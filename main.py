import argparse
import pprint
from src.routing.agent import DRLAgent
from src.metrics.evaluator import SystemEvaluator
from src.metrics.visualization import Plotter
from src.metrics.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="6G-Oriented QoS-Aware DRL Routing in LEO Constellations")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'verify'], default='train',
                        help='train: PPO training | eval: compare DRL vs baselines | verify: quick routing sanity check')
    parser.add_argument('--resume', action='store_true', help='Resume training from existing checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    args = parser.parse_args()
    
    agent = DRLAgent()
    
    if args.mode == 'verify':
        agent.check_environment()
        agent.quick_verify()
    elif args.mode == 'train':
        agent.check_environment()
        agent.train(resume=args.resume)
    elif args.mode == 'eval':
        evaluator = SystemEvaluator(agent.env, agent.topology, agent.model_path)
        plotter = Plotter(output_dir="outputs")
        report_gen = ReportGenerator(output_dir="outputs")
        
        methods = ['drl', 'dijkstra', 'ospf']
        results = {}
        
        for method in methods:
            metrics = evaluator.run_evaluation(routing_method=method)
            results[method] = metrics
            print(f"\n--- Results for {method.upper()} ---")
            pprint.pprint(metrics)
            
        print("\nGenerating comparison plots...")
        slice_order = ['urllc', 'embb', 'mmtc']
        
        # Basic comparison plots
        plotter.plot_bar_comparison(results, 'pdr', 'Packet Delivery Ratio (%)', 'PDR (%)', 'pdr_comparison.png')
        plotter.plot_bar_comparison(results, 'avg_bandwidth_utilization', 'Average Link Utilization (%)', 'Utilization (%)', 'utilization_comparison.png')
        plotter.plot_bar_comparison(results, 'avg_hop_count', 'Average Route Hop Count', 'Hop Count (Nodes)', 'hop_count_comparison.png')
        plotter.plot_bar_comparison(results, 'routing_loops', 'Routing Loops Detected', 'Loop Count', 'routing_loops_comparison.png')
        
        # Slice-wise comparisons
        plotter.plot_grouped_bar_comparison(results, 'latency_ms', 'Average Latency by Slice (ms)', 'Latency (ms)', 'latency_comparison.png', slice_order=slice_order)
        plotter.plot_grouped_bar_comparison(results, 'latency_p95_ms', 'P95 Latency by Slice (ms)', 'P95 Latency (ms)', 'latency_p95_comparison.png', slice_order=slice_order)
        plotter.plot_grouped_bar_comparison(results, 'jitter_ms', 'Average Jitter by Slice (ms)', 'Jitter (ms)', 'jitter_comparison.png', slice_order=slice_order)
        plotter.plot_grouped_bar_comparison(results, 'throughput_mbps', 'Average Throughput by Slice (Mbps)', 'Throughput (Mbps)', 'throughput_comparison.png', slice_order=slice_order)
        plotter.plot_grouped_bar_comparison(results, 'qos_satisfaction', 'QoS Satisfaction Rate by Slice (%)', 'Satisfaction (%)', 'qos_comparison.png', slice_order=slice_order)
        
        # Enhanced plots
        plotter.plot_drop_reasons(results)
        plotter.plot_latency_distribution(results)
        plotter.plot_comprehensive_comparison(results)
        plotter.plot_training_progress()
        
        # Generate reports
        report_gen.generate_summary(results)
        report_gen.generate_json_report(results)
        
        print("\nAll plots and reports saved to 'outputs/' directory.")

if __name__ == "__main__":
    main()
