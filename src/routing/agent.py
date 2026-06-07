import os
import sys
import yaml
import torch
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
            writer.writerow(['step', 'total_delivered', 'total_dropped', 'dropped_loops', 'dropped_max_hops', 
                             'dropped_invalid', 'dropped_congestion', 'dropped_qos_timeout',
                             'backtrack_count', 'routing_loops_detected',
                             'ep_reward_delivery', 'ep_reward_qos', 'ep_reward_latency_shaping', 'ep_reward_progress', 
                             'ep_penalty_loops', 'ep_penalty_congestion', 'ep_penalty_hops', 'ep_penalty_backtrack'])

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
                            info.get('dropped_congestion', 0),
                            info.get('dropped_qos_timeout', 0),
                            info.get('backtrack_count', 0),
                            info.get('routing_loops_detected', 0),
                            info.get('ep_reward_delivery', 0),
                            info.get('ep_reward_qos', 0),
                            info.get('ep_reward_latency_shaping', 0),
                            info.get('ep_reward_progress', 0),
                            info.get('ep_penalty_loops', 0),
                            info.get('ep_penalty_congestion', 0),
                            info.get('ep_penalty_hops', 0),
                            info.get('ep_penalty_backtrack', 0)
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
                    self.logger.record("networking/dropped_congestion", info.get('dropped_congestion', 0))
                    self.logger.record("networking/dropped_qos_timeout", info.get('dropped_qos_timeout', 0))
                    self.logger.record("networking/backtrack_count", info.get('backtrack_count', 0))
                    self.logger.record("networking/routing_loops", info.get('routing_loops_detected', 0))
                    self.logger.record("reward_components/delivery", info.get('ep_reward_delivery', 0))
                    self.logger.record("reward_components/qos", info.get('ep_reward_qos', 0))
                    self.logger.record("reward_components/latency_shaping", info.get('ep_reward_latency_shaping', 0))
                    self.logger.record("reward_components/progress", info.get('ep_reward_progress', 0))
                    self.logger.record("reward_components/penalty_loops", info.get('ep_penalty_loops', 0))
                    self.logger.record("reward_components/penalty_congestion", info.get('ep_penalty_congestion', 0))
                    self.logger.record("reward_components/penalty_hops", info.get('ep_penalty_hops', 0))
                    self.logger.record("reward_components/penalty_backtrack", info.get('ep_penalty_backtrack', 0))
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
        
        self._init_vec_normalize()
        
    def _init_vec_normalize(self):
        """Load VecNormalize only when compatible with current observation/action space."""
        if os.path.exists(self.vec_path):
            try:
                loaded = VecNormalize.load(self.vec_path, self.env)
                if loaded.observation_space.shape != self.env.observation_space.shape:
                    raise ValueError("observation space changed")
                if loaded.action_space.n != self.env.action_space.n:
                    raise ValueError("action space changed (retrain required)")
                self.env = loaded
                self.env.training = True
                print("Loaded existing VecNormalize stats.")
                return
            except (AssertionError, ValueError, AttributeError) as e:
                print(f"Resetting VecNormalize stats ({e}).")
        self.env = VecNormalize(
            self.env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
        )

    def check_environment(self):
        print("Checking custom Gym environment compliance...")
        check_env(self.raw_env, warn=True)
        print("Environment check passed.")

    def train(self, resume=False):
        drl_config = self.config['drl']
        from stable_baselines3.common.callbacks import EvalCallback

        if os.path.exists(self.model_path) and not resume:
            print(f"Removing stale model at {self.model_path} (action/obs space changed).")
            os.remove(self.model_path)
        if os.path.exists(self.vec_path) and not resume:
            os.remove(self.vec_path)
            self._init_vec_normalize()

        self.raw_env.drl_training = True
        self.raw_env.drl_use_supervisor = False

        print(f"Starting {drl_config['algorithm']} training for {drl_config['total_timesteps']} timesteps...")
        print("Action space: 0=Dijkstra hop, 1=OSPF hop (meta-routing), obs space includes prediction features")

        eval_callback = EvalCallback(
            self.env,
            best_model_save_path='models/',
            log_path='training_logs/',
            eval_freq=20000,
            n_eval_episodes=3,
            deterministic=True,
            render=False
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing PPO Model on device: {device.upper()}")

        if resume and os.path.exists(self.model_path):
            model = PPO.load(self.model_path, env=self.env, device=device)
            print(f"Resuming training from {self.model_path}")
        else:
            model = None

        if model is None:
            model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=drl_config['learning_rate'],
                n_steps=drl_config['n_steps'],
                batch_size=drl_config['batch_size'],
                n_epochs=drl_config['n_epochs'],
                gamma=drl_config['gamma'],
                gae_lambda=drl_config.get('gae_lambda', 0.98),
                ent_coef=drl_config.get('ent_coef', 0.005),
                clip_range=drl_config.get('clip_range', 0.15),
                vf_coef=drl_config.get('vf_coef', 0.7),
                max_grad_norm=drl_config.get('max_grad_norm', 0.5),
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                device=device,
            )

        new_logger = configure("./training_logs/", ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        # Combine callbacks
        from stable_baselines3.common.callbacks import CallbackList
        metrics_callback = NetworkingMetricsCallback()
        callbacks = CallbackList([eval_callback, metrics_callback])
        
        model.learn(total_timesteps=drl_config['total_timesteps'], callback=callbacks, reset_num_timesteps=not resume)
        
        # Save final model
        print(f"\nTraining complete! Saving final model...")
        model.save(self.model_path)
        self.env.save(self.vec_path)
        print(f"✓ Final model saved to {self.model_path}")
        print(f"✓ VecNormalize stats saved to {self.vec_path}")
        
        # Check if best_model exists
        best_model_path = "models/best_model.zip"
        if os.path.exists(best_model_path):
            print(f"✓ Best model saved to {best_model_path}")
        
        print(f"\nTo evaluate: python main.py --mode eval")

    def quick_verify(self, episodes=1):
        """Compare baselines vs DRL meta-router (supervisor on, no checkpoint needed)."""
        from src.metrics.evaluator import SystemEvaluator

        print("Quick verification (meta-DRL with cost supervisor)...")
        evaluator = SystemEvaluator(self.env, self.topology, self.model_path)
        results = {}
        for method in ('dijkstra', 'ospf', 'drl'):
            results[method] = evaluator.run_evaluation(routing_method=method)
            m = results[method]
            print(
                f"  {method.upper():8s} PDR={m['pdr']:5.1f}%  "
                f"latency={m['avg_latency']:5.1f}ms  loops={m['routing_loops']}"
            )
        d_pdr = results['drl']['pdr']
        d_lat = results['drl']['avg_latency']
        print(
            f"\nDRL vs Dijkstra  PDR {d_pdr - results['dijkstra']['pdr']:+.1f}%  "
            f"latency {d_lat - results['dijkstra']['avg_latency']:+.1f} ms"
        )
        print(
            f"DRL vs OSPF      PDR {d_pdr - results['ospf']['pdr']:+.1f}%  "
            f"latency {d_lat - results['ospf']['avg_latency']:+.1f} ms"
        )
        ok = d_pdr >= max(results['dijkstra']['pdr'], results['ospf']['pdr'])
        print("PASS: DRL best PDR" if ok else "FAIL: DRL should beat both baselines — check config")
        print("Retrain: python main.py --mode train")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DRL Satellite Routing Agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'check'],
                        help='Mode: train or check environment. Use main.py --mode eval for full evaluation.')
    args = parser.parse_args()
    
    agent = DRLAgent()
    
    if args.mode == 'check':
        agent.check_environment()
    elif args.mode == 'train':
        agent.train()
