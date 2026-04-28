import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_bar_comparison(self, results, metric_key, title, ylabel, filename):
        """
        results: dict { 'DRL': {metric_key: val}, 'Dijkstra': {metric_key: val}, ... }
        """
        methods = list(results.keys())
        values = [results[m].get(metric_key, 0) for m in methods]
        
        plt.figure(figsize=(10, 6))
        plt.bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_grouped_bar_comparison(self, results, metric_key, title, ylabel, filename,
                                    slice_order=None):
        """
        results: dict { 'drl': {metric_key: {'urllc': val, 'embb': val}}, ... }
        slice_order: optional list to fix the display order of slices (e.g. ['urllc', 'embb', 'mmtc'])
        """
        methods = list(results.keys())
        if not methods: return
        
        # Collect sub-keys from ALL methods so no slice disappears when one
        # method delivers zero packets for that slice type.
        all_sub_keys = set()
        for m in methods:
            all_sub_keys.update(results[m].get(metric_key, {}).keys())
        
        if slice_order:
            # Keep only keys that exist, preserving specified order, then append any extras
            sub_keys = [s for s in slice_order if s in all_sub_keys]
            sub_keys += sorted(all_sub_keys - set(sub_keys))
        else:
            sub_keys = sorted(all_sub_keys)
        
        if not sub_keys: return
        
        x = np.arange(len(sub_keys))
        width = 0.25  # the width of the bars
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, method in enumerate(methods):
            method_sub_dict = results[method].get(metric_key, {})
            # Default to 0 for any slice the method has no data for
            vals = [method_sub_dict.get(sk, 0) for sk in sub_keys]
            offset = (i - len(methods)/2) * width + width/2
            ax.bar(x + offset, vals, width, label=method)
            
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sub_keys)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
