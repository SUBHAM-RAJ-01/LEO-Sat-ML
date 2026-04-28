import os
import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import csv
from stable_baselines3.common.callbacks import BaseCallback
from src.env.topology import SatelliteNetwork
from src.traffic.generator import TrafficGenerator
from src.routing.drl_env import SatelliteRoutingEnv

class NetworkingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(NetworkingMetricsCallback, self).__init__(verbose)
        self.csv_path = "training_logs/training_metrics.csv"
        os.makedirs("training_logs", exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'total_delivered', 'total_dropped', 'dropped_loops', 'dropped_max_hops', 'dropped_invalid', 
                             'backtrack_count',
                             'ep_reward_delivery', 'ep_reward_qos', 'ep_reward_progress', 
                             'ep_reward_urgency', 'ep_penalty_loops', 'ep_penalty_congestion'])

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            infos = self.locals.get("infos")
            if infos:
                info = infos[0]
                if 'ep_reward_delivery' in info:
                    # Log to CSV
                    with open(self.csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.num_timesteps,
                            info.get('total_delivered', 0),
                            info.get('total_dropped', 0),
                            info.get('dropped_loops', 0),
                            info.get('dropped_max_hops', 0),
                            info.get('dropped_invalid', 0),
                            info.get('backtrack_count', 0),
                            info.get('ep_reward_delivery', 0),
                            info.get('ep_reward_qos', 0),
                            info.get('ep_reward_progress', 0),
                            info.get('ep_reward_urgency', 0),
                            info.get('ep_penalty_loops', 0),
                            info.get('ep_penalty_congestion', 0)
                        ])
                    
                    # Log to TensorBoard
                    dels = info.get('total_delivered', 0)
                    drops = info.get('total_dropped', 0)
                    total = dels + drops
                    pdr = (dels / total * 100.0) if total > 0 else 0.0
                    self.logger.record("networking/pdr_percent", pdr)
                    self.logger.record("networking/total_delivered", dels)
                    self.logger.record("networking/total_dropped", drops)
                    self.logger.record("networking/dropped_loops", info.get('dropped_loops', 0))
                    self.logger.record("networking/dropped_max_hops", info.get('dropped_max_hops', 0))
                    self.logger.record("networking/backtrack_count", info.get('backtrack_count', 0))
                    self.logger.record("reward_components/delivery", info.get('ep_reward_delivery', 0))
                    self.logger.record("reward_components/qos", info.get('ep_reward_qos', 0))
                    self.logger.record("reward_components/progress", info.get('ep_reward_progress', 0))
                    self.logger.record("reward_components/urgency", info.get('ep_reward_urgency', 0))
                    self.logger.record("reward_components/penalty_loops", info.get('ep_penalty_loops', 0))
                    self.logger.record("reward_components/penalty_congestion", info.get('ep_penalty_congestion', 0))
        return True

class DRLAgent:
    def __init__(self, config_path="config/config.yaml", model_path="models/ppo_routing.zip"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.topology = SatelliteNetwork(config_path)
        self.num_nodes = len(self.topology.satellites)
        self.traffic_gen = TrafficGenerator(self.config, self.num_nodes)
        
        self.raw_env = SatelliteRoutingEnv(self.topology, self.traffic_gen, config_path)
        
        # SB3 Monitor to track accurate episodic rewards (ep_rew_mean)
        log_dir = "training_logs"
        os.makedirs(log_dir, exist_ok=True)
        monitored_env = Monitor(self.raw_env, filename=os.path.join(log_dir, "ppo_monitor"))
        
        self.env = DummyVecEnv([lambda: monitored_env])
        
        self.model_path = model_path
        self.vec_path = model_path.replace('.zip', '_vecnormalize.pkl')
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        if os.path.exists(self.vec_path):
            try:
                self.env = VecNormalize.load(self.vec_path, self.env)
                self.env.training = True
                print("Loaded existing VecNormalize stats.")
            except (AssertionError, ValueError, AttributeError) as e:
                print(f"Warning: Could not load VecNormalize stats from {self.vec_path} due to shape/version mismatch ({e}). Resetting stats.")
                self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
        else:
            self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
        
    def check_environment(self):
        print("Checking custom Gym environment compliance...")
        check_env(self.raw_env, warn=True)
        print("Environment check passed.")

    def train(self):
        drl_config = self.config['drl']
        from stable_baselines3.common.callbacks import EvalCallback
        
        print(f"Starting {drl_config['algorithm']} training for {drl_config['total_timesteps']} timesteps...")
        
        eval_callback = EvalCallback(
            self.env, 
            best_model_save_path='models/',
            log_path='training_logs/',
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing PPO Model on device: {device.upper()}")
        
        model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=drl_config['learning_rate'],
            n_steps=drl_config['n_steps'],
            batch_size=drl_config['batch_size'],
            n_epochs=drl_config['n_epochs'],
            gamma=drl_config['gamma'],
            gae_lambda=drl_config.get('gae_lambda', 0.95),
            ent_coef=drl_config['ent_coef'],
            max_grad_norm=drl_config.get('max_grad_norm', 0.5),
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            device=device
        )
        
        new_logger = configure("./training_logs/", ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        # Combine callbacks
        from stable_baselines3.common.callbacks import CallbackList
        metrics_callback = NetworkingMetricsCallback()
        callbacks = CallbackList([eval_callback, metrics_callback])
        
        model.learn(total_timesteps=drl_config['total_timesteps'], callback=callbacks)
        model.save(self.model_path)
        self.env.save(self.vec_path)
        print(f"Model saved to {self.model_path} and VecNormalize stats to {self.vec_path}")
        
    def evaluate(self, num_episodes=5):
        print(f"Evaluating the trained model on {num_episodes} episodes...")
        model = PPO.load(self.model_path)
        
        all_rewards = []
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            
            # Simple loop until simulation timestep limit is reached
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                
            all_rewards.append(total_reward)
            print(f"Episode {episode + 1} finalized. Reward: {total_reward}, Delivered: {len(self.env.delivered_packets)}, Dropped: {self.env.dropped_packets}")
            
        print(f"Average evaluation reward: {sum(all_rewards) / num_episodes}")

if __name__ == "__main__":
    agent = DRLAgent()
    # Check env config
    # agent.check_environment()
    agent.train()
    # agent.evaluate()
