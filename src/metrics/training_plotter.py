import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class TrainingPlotter:
    """Generate clean training plots for reports and presentations"""
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set clean style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
    
    def plot_training_metrics(self, csv_path="training_logs/progress.csv"):
        """Generate comprehensive training metrics plot"""
        
        if not os.path.exists(csv_path):
            print(f"Training log not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
        
        # 1. Episode Reward
        ax1 = axes[0, 0]
        if 'rollout/ep_rew_mean' in df.columns:
            ax1.plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], 
                    linewidth=2, color='#2E86AB', label='Episode Reward')
            ax1.fill_between(df['time/total_timesteps'], 
                            df['rollout/ep_rew_mean'] * 0.95,
                            df['rollout/ep_rew_mean'] * 1.05,
                            alpha=0.2, color='#2E86AB')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Mean Episode Reward')
            ax1.set_title('Episode Reward Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. PDR (Packet Delivery Ratio)
        ax2 = axes[0, 1]
        if 'networking/pdr_percent' in df.columns:
            ax2.plot(df['time/total_timesteps'], df['networking/pdr_percent'],
                    linewidth=2, color='#27AE60', label='PDR')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Target (80%)')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('PDR (%)')
            ax2.set_title('Packet Delivery Ratio')
            ax2.set_ylim([0, 100])
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. Routing Loops
        ax3 = axes[1, 0]
        if 'networking/routing_loops' in df.columns:
            ax3.plot(df['time/total_timesteps'], df['networking/routing_loops'],
                    linewidth=2, color='#E74C3C', label='Routing Loops')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Loop Count')
            ax3.set_title('Routing Loops Detected')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Training Loss
        ax4 = axes[1, 1]
        if 'train/loss' in df.columns:
            ax4.plot(df['time/total_timesteps'], df['train/loss'],
                    linewidth=2, color='#9B59B6', label='Total Loss')
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Loss')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'training_metrics.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"Training metrics plot saved: {output_path}")
    
    def plot_reward_components(self, csv_path="training_logs/training_metrics.csv"):
        """Plot reward components breakdown"""
        
        if not os.path.exists(csv_path):
            print(f"Training metrics not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot positive rewards
        if 'ep_reward_delivery' in df.columns:
            ax.plot(df['step'], df['ep_reward_delivery'], 
                   linewidth=2, label='Delivery Reward', color='#27AE60')
        if 'ep_reward_progress' in df.columns:
            ax.plot(df['step'], df['ep_reward_progress'],
                   linewidth=2, label='Progress Reward', color='#3498DB')
        if 'ep_reward_qos' in df.columns:
            ax.plot(df['step'], df['ep_reward_qos'],
                   linewidth=2, label='QoS Reward', color='#F39C12')
        
        # Plot penalties (negative)
        if 'ep_penalty_loops' in df.columns:
            ax.plot(df['step'], df['ep_penalty_loops'],
                   linewidth=2, label='Loop Penalty', color='#E74C3C', linestyle='--')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Reward Value')
        ax.set_title('Reward Components Over Training')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'reward_components.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"Reward components plot saved: {output_path}")
    
    def plot_network_metrics(self, csv_path="training_logs/training_metrics.csv"):
        """Plot network-specific metrics"""
        
        if not os.path.exists(csv_path):
            return
        
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Network Performance Metrics', fontsize=14, fontweight='bold')
        
        # Delivered vs Dropped
        ax1 = axes[0]
        if 'total_delivered' in df.columns and 'total_dropped' in df.columns:
            ax1.plot(df['step'], df['total_delivered'], 
                    linewidth=2, label='Delivered', color='#27AE60')
            ax1.plot(df['step'], df['total_dropped'],
                    linewidth=2, label='Dropped', color='#E74C3C')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Packet Count')
            ax1.set_title('Packets Delivered vs Dropped')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Drop reasons breakdown
        ax2 = axes[1]
        if 'dropped_loops' in df.columns:
            ax2.plot(df['step'], df['dropped_loops'],
                    linewidth=2, label='Loops', color='#E74C3C')
        if 'dropped_max_hops' in df.columns:
            ax2.plot(df['step'], df['dropped_max_hops'],
                    linewidth=2, label='Max Hops', color='#F39C12')
        if 'dropped_qos_timeout' in df.columns:
            ax2.plot(df['step'], df['dropped_qos_timeout'],
                    linewidth=2, label='QoS Timeout', color='#9B59B6')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Drop Count')
        ax2.set_title('Drop Reasons Breakdown')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PDR trend
        ax3 = axes[2]
        if 'total_delivered' in df.columns and 'total_dropped' in df.columns:
            pdr = (df['total_delivered'] / (df['total_delivered'] + df['total_dropped'])) * 100
            ax3.plot(df['step'], pdr, linewidth=2, color='#2E86AB')
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Target')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('PDR (%)')
            ax3.set_title('PDR Trend')
            ax3.set_ylim([0, 100])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'network_metrics.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"Network metrics plot saved: {output_path}")
