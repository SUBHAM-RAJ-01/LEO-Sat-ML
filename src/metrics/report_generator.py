import os
import json
from datetime import datetime

class ReportGenerator:
    """Generate clean summary reports for presentations and documentation"""
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_summary(self, results, training_metrics=None):
        """Generate a clean text summary of evaluation results"""
        
        summary_path = os.path.join(self.output_dir, "evaluation_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EVALUATION RESULTS SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall comparison
            f.write("OVERALL PERFORMANCE COMPARISON\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Metric':<25} {'DRL':<15} {'DIJKSTRA':<15} {'OSPF':<15}\n")
            f.write("-" * 70 + "\n")
            
            metrics_to_show = [
                ('PDR (%)', 'pdr'),
                ('Avg Latency (ms)', 'avg_latency'),
                ('P95 Latency (ms)', 'p95_latency'),
                ('Routing Loops', 'routing_loops'),
                ('Avg Hop Count', 'avg_hop_count'),
                ('Delivered Packets', 'total_delivered'),
                ('Dropped Packets', 'total_dropped')
            ]
            
            for label, key in metrics_to_show:
                drl_val = results.get('drl', {}).get(key, 0)
                dij_val = results.get('dijkstra', {}).get(key, 0)
                ospf_val = results.get('ospf', {}).get(key, 0)
                
                f.write(f"{label:<25} {drl_val:<15.2f} {dij_val:<15.2f} {ospf_val:<15.2f}\n")
            
            f.write("\n")
            
            # QoS by slice
            f.write("QoS SATISFACTION BY SLICE (%)\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Slice':<25} {'DRL':<15} {'DIJKSTRA':<15} {'OSPF':<15}\n")
            f.write("-" * 70 + "\n")
            
            for slice_type in ['urllc', 'embb', 'mmtc']:
                drl_qos = results.get('drl', {}).get('qos_satisfaction', {}).get(slice_type, 0)
                dij_qos = results.get('dijkstra', {}).get('qos_satisfaction', {}).get(slice_type, 0)
                ospf_qos = results.get('ospf', {}).get('qos_satisfaction', {}).get(slice_type, 0)
                
                f.write(f"{slice_type.upper():<25} {drl_qos:<15.2f} {dij_qos:<15.2f} {ospf_qos:<15.2f}\n")
            
            f.write("\n")
            
            # Drop reasons
            f.write("DROP REASONS BREAKDOWN\n")
            f.write("-" * 70 + "\n")
            for method in ['drl', 'dijkstra', 'ospf']:
                f.write(f"\n{method.upper()}:\n")
                drop_reasons = results.get(method, {}).get('drop_reasons', {})
                for reason, count in drop_reasons.items():
                    f.write(f"  {reason:<20}: {count:>5}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Summary saved to: {summary_path}")
        return summary_path
    
    def generate_json_report(self, results):
        """Save results as JSON for further processing"""
        json_path = os.path.join(self.output_dir, "evaluation_results.json")
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"JSON results saved to: {json_path}")
        return json_path
