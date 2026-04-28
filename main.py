import argparse
import pprint
from src.routing.agent import DRLAgent
from src.metrics.evaluator import SystemEvaluator
from src.metrics.visualization import Plotter

def main():
    parser = argparse.ArgumentParser(description="6G-Oriented QoS-Aware DRL Routing in LEO Constellations")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', 
                        help='Mode to run: train the PPO agent or evaluate the system against baselines.')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    args = parser.parse_args()
    
    agent = DRLAgent()
    
    if args.mode == 'train':
        agent.check_environment()
        agent.train()
    elif args.mode == 'eval':
        evaluator = SystemEvaluator(agent.env, agent.topology, agent.model_path)
        plotter = Plotter(output_dir="outputs")
        
        methods = ['drl', 'dijkstra', 'ospf']
        results = {}
        
        for method in methods:
            metrics = evaluator.run_evaluation(routing_method=method)
            results[method] = metrics
            print(f"\n--- Results for {method.upper()} ---")
            pprint.pprint(metrics)
            
        print("\nGenerating comparison plots in 'outputs' directory...")
        slice_order = ['urllc', 'embb', 'mmtc']
        plotter.plot_bar_comparison(results, 'pdr', 'Packet Delivery Ratio (%)', 'PDR (%)', 'pdr_comparison.png')
        plotter.plot_bar_comparison(results, 'avg_bandwidth_utilization', 'Average Link Utilization (%)', 'Utilization (%)', 'utilization_comparison.png')
        plotter.plot_bar_comparison(results, 'avg_hop_count', 'Average Route Hop Count', 'Hop Count (Nodes)', 'hop_count_comparison.png')
        plotter.plot_grouped_bar_comparison(results, 'latency_ms', 'Average Latency by Slice (ms)', 'Latency (ms)', 'latency_comparison.png', slice_order=slice_order)
        plotter.plot_grouped_bar_comparison(results, 'jitter_ms', 'Average Jitter by Slice (ms)', 'Jitter (ms)', 'jitter_comparison.png', slice_order=slice_order)
        plotter.plot_grouped_bar_comparison(results, 'throughput_mbps', 'Average Throughput by Slice (Mbps)', 'Throughput (Mbps)', 'throughput_comparison.png', slice_order=slice_order)
        plotter.plot_grouped_bar_comparison(results, 'qos_satisfaction', 'QoS Satisfaction Rate by Slice (%)', 'Satisfaction (%)', 'qos_comparison.png', slice_order=slice_order)
        print("Done!")

if __name__ == "__main__":
    main()
