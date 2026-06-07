import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class Plotter:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        
    def plot_bar_comparison(self, results, metric_key, title, ylabel, filename):
        """
        results: dict { 'DRL': {metric_key: val}, 'Dijkstra': {metric_key: val}, ... }
        """
        methods = list(results.keys())
        values = [results[m].get(metric_key, 0) for m in methods]
        
        plt.figure(figsize=(10, 6))
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = plt.bar(methods, values, color=colors[:len(methods)], edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.title(title, fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()

    def plot_grouped_bar_comparison(self, results, metric_key, title, ylabel, filename,
                                    slice_order=None):
        """
        results: dict { 'drl': {metric_key: {'urllc': val, 'embb': val}}, ... }
        slice_order: optional list to fix the display order of slices (e.g. ['urllc', 'embb', 'mmtc'])
        """
        methods = list(results.keys())
        if not methods: return
        
        # Collect sub-keys from ALL methods
        all_sub_keys = set()
        for m in methods:
            all_sub_keys.update(results[m].get(metric_key, {}).keys())
        
        if slice_order:
            sub_keys = [s for s in slice_order if s in all_sub_keys]
            sub_keys += sorted(all_sub_keys - set(sub_keys))
        else:
            sub_keys = sorted(all_sub_keys)
        
        if not sub_keys: return
        
        x = np.arange(len(sub_keys))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, method in enumerate(methods):
            method_sub_dict = results[method].get(metric_key, {})
            vals = [method_sub_dict.get(sk, 0) for sk in sub_keys]
            offset = (i - len(methods)/2) * width + width/2
            bars = ax.bar(x + offset, vals, width, label=method.upper(), 
                         color=colors[i % len(colors)], edgecolor='black', linewidth=1.0)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=8)
            
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in sub_keys])
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
    
    def plot_drop_reasons(self, results, filename='drop_reasons_comparison.png'):
        """Plot drop reasons comparison across methods"""
        methods = list(results.keys())
        drop_categories = ['loops', 'max_hops', 'invalid', 'congestion', 'qos_timeout']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(drop_categories))
        width = 0.25
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, method in enumerate(methods):
            drop_reasons = results[method].get('drop_reasons', {})
            vals = [drop_reasons.get(cat, 0) for cat in drop_categories]
            offset = (i - len(methods)/2) * width + width/2
            ax.bar(x + offset, vals, width, label=method.upper(), 
                  color=colors[i % len(colors)], edgecolor='black', linewidth=1.0)
        
        ax.set_ylabel('Number of Dropped Packets', fontweight='bold')
        ax.set_title('Packet Drop Reasons Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in drop_categories])
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
    
    def plot_training_progress(self, csv_path='training_logs/training_metrics.csv', 
                              filename='training_progress.png'):
        """Plot training progress from CSV logs"""
        if not os.path.exists(csv_path):
            print(f"Training metrics CSV not found at {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # PDR over time
        if 'total_delivered' in df.columns and 'total_dropped' in df.columns:
            total = df['total_delivered'] + df['total_dropped']
            pdr = (df['total_delivered'] / total.replace(0, 1)) * 100
            axes[0, 0].plot(df['step'], pdr, linewidth=2, color='#2E86AB')
            axes[0, 0].set_xlabel('Training Steps', fontweight='bold')
            axes[0, 0].set_ylabel('PDR (%)', fontweight='bold')
            axes[0, 0].set_title('Packet Delivery Ratio Over Training', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Routing loops over time
        if 'routing_loops_detected' in df.columns:
            axes[0, 1].plot(df['step'], df['routing_loops_detected'], 
                           linewidth=2, color='#A23B72')
            axes[0, 1].set_xlabel('Training Steps', fontweight='bold')
            axes[0, 1].set_ylabel('Routing Loops', fontweight='bold')
            axes[0, 1].set_title('Routing Loops Over Training', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Reward components
        reward_cols = ['ep_reward_delivery', 'ep_reward_qos', 'ep_reward_latency_shaping']
        for col in reward_cols:
            if col in df.columns:
                label = col.replace('ep_reward_', '').replace('_', ' ').title()
                axes[1, 0].plot(df['step'], df[col], linewidth=1.5, label=label, alpha=0.8)
        axes[1, 0].set_xlabel('Training Steps', fontweight='bold')
        axes[1, 0].set_ylabel('Reward', fontweight='bold')
        axes[1, 0].set_title('Reward Components Over Training', fontweight='bold')
        axes[1, 0].legend(loc='best')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Penalty components
        penalty_cols = ['ep_penalty_loops', 'ep_penalty_congestion', 'ep_penalty_hops']
        for col in penalty_cols:
            if col in df.columns:
                label = col.replace('ep_penalty_', '').replace('_', ' ').title()
                axes[1, 1].plot(df['step'], df[col], linewidth=1.5, label=label, alpha=0.8)
        axes[1, 1].set_xlabel('Training Steps', fontweight='bold')
        axes[1, 1].set_ylabel('Penalty', fontweight='bold')
        axes[1, 1].set_title('Penalty Components Over Training', fontweight='bold')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
        print(f"Training progress plot saved to {os.path.join(self.output_dir, filename)}")
    
    def plot_latency_distribution(self, results, filename='latency_distribution.png'):
        """Plot latency distribution comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(results.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, method in enumerate(methods):
            avg_lat = results[method].get('avg_latency', 0)
            p95_lat = results[method].get('p95_latency', 0)
            
            # Plot average with full opacity
            ax.bar(i*3, avg_lat, 
                  color=colors[i % len(colors)],
                  alpha=1.0, edgecolor='black', linewidth=1.0,
                  label=method.upper())
            
            # Plot P95 with reduced opacity
            ax.bar(i*3+1, p95_lat, 
                  color=colors[i % len(colors)],
                  alpha=0.6, edgecolor='black', linewidth=1.0)
            
            # Add labels
            ax.text(i*3, avg_lat, f'{avg_lat:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(i*3+1, p95_lat, f'{p95_lat:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Set x-axis labels
        x_labels = []
        for method in methods:
            x_labels.extend([f'{method.upper()}\nAvg', f'{method.upper()}\nP95'])
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=9)
        
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.set_title('Latency Distribution: Average vs P95', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
    
    def plot_comprehensive_comparison(self, results, filename='comprehensive_comparison.png'):
        """Create a comprehensive multi-metric comparison"""
        methods = list(results.keys())
        metrics = ['pdr', 'avg_latency', 'avg_hop_count', 'routing_loops']
        metric_labels = ['PDR (%)', 'Avg Latency (ms)', 'Avg Hop Count', 'Routing Loops']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [results[m].get(metric, 0) for m in methods]
            bars = axes[idx].bar(methods, values, color=colors[:len(methods)], 
                                edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            axes[idx].set_ylabel(label, fontweight='bold')
            axes[idx].set_title(label, fontweight='bold')
            axes[idx].grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
