"""
Visualization tools for traffic prediction performance evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


class PredictionVisualizer:
    """Generates publication-quality plots for traffic prediction evaluation."""
    
    def __init__(self, output_dir='outputs/prediction'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_prediction_vs_actual(self, actual, predicted_lstm, predicted_gru, 
                                   timestamps=None, metric_name='Queue Length',
                                   output_filename='prediction_comparison.png',
                                   figsize=(14, 6), dpi=300):
        """
        Plot actual vs predicted traffic for LSTM and GRU models.
        
        Args:
            actual: Actual values (timesteps,)
            predicted_lstm: LSTM predictions
            predicted_gru: GRU predictions
            timestamps: Optional time axis
            metric_name: Y-axis label
        """
        if timestamps is None:
            timestamps = np.arange(len(actual))
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # LSTM comparison
        axes[0].plot(timestamps, actual, label='Actual', color='black', 
                    linewidth=2, alpha=0.7)
        axes[0].plot(timestamps, predicted_lstm, label='LSTM Prediction', 
                    color='blue', linewidth=1.5, linestyle='--', alpha=0.8)
        axes[0].fill_between(timestamps, actual, predicted_lstm, alpha=0.2, color='blue')
        axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel(metric_name, fontsize=12)
        axes[0].set_title('LSTM Traffic Prediction', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # GRU comparison
        axes[1].plot(timestamps, actual, label='Actual', color='black', 
                    linewidth=2, alpha=0.7)
        axes[1].plot(timestamps, predicted_gru, label='GRU Prediction', 
                    color='green', linewidth=1.5, linestyle='--', alpha=0.8)
        axes[1].fill_between(timestamps, actual, predicted_gru, alpha=0.2, color='green')
        axes[1].set_xlabel('Time Step', fontsize=12)
        axes[1].set_ylabel(metric_name, fontsize=12)
        axes[1].set_title('GRU Traffic Prediction', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Prediction comparison saved to {filepath}")
        return filepath
    
    def plot_multi_horizon_forecast(self, actual, predictions_dict, timestamps=None,
                                    metric_name='Congestion Level (%)',
                                    output_filename='multi_horizon_forecast.png',
                                    figsize=(12, 7), dpi=300):
        """
        Plot multi-step-ahead forecasts.
        
        Args:
            actual: Ground truth (timesteps,)
            predictions_dict: {horizon: predictions} e.g., {5: pred_5step, 10: pred_10step}
        """
        if timestamps is None:
            timestamps = np.arange(len(actual))
        
        plt.figure(figsize=figsize, dpi=dpi)
        
        plt.plot(timestamps, actual, label='Actual', color='black', 
                linewidth=2.5, alpha=0.8, zorder=10)
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for idx, (horizon, predictions) in enumerate(predictions_dict.items()):
            color = colors[idx % len(colors)]
            plt.plot(timestamps[:len(predictions)], predictions, 
                    label=f'{horizon}-Step Ahead', color=color,
                    linewidth=1.5, linestyle='--', alpha=0.7)
        
        plt.xlabel('Time Step', fontsize=13)
        plt.ylabel(metric_name, fontsize=13)
        plt.title('Multi-Horizon Traffic Forecasting', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Multi-horizon forecast saved to {filepath}")
        return filepath
    
    def plot_error_distribution(self, errors_lstm, errors_gru, metric_name='Prediction Error',
                               output_filename='error_distribution.png',
                               figsize=(12, 5), dpi=300):
        """Plot error distribution histograms for LSTM and GRU."""
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # LSTM errors
        axes[0].hist(errors_lstm.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel(metric_name, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'LSTM Error Distribution (μ={np.mean(errors_lstm):.3f})', 
                         fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # GRU errors
        axes[1].hist(errors_gru.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[1].set_xlabel(metric_name, fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'GRU Error Distribution (μ={np.mean(errors_gru):.3f})', 
                         fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Error distribution saved to {filepath}")
        return filepath
    
    def plot_metrics_comparison(self, metrics_lstm, metrics_gru, 
                               output_filename='metrics_comparison.png',
                               figsize=(10, 6), dpi=300):
        """
        Bar chart comparing LSTM vs GRU metrics (MAE, RMSE, MAPE).
        
        Args:
            metrics_lstm: dict with 'mae', 'rmse', 'mape'
            metrics_gru: dict with 'mae', 'rmse', 'mape'
        """
        metrics_names = ['MAE', 'RMSE', 'MAPE (%)']
        lstm_values = [metrics_lstm['mae'], metrics_lstm['rmse'], metrics_lstm['mape']]
        gru_values = [metrics_gru['mae'], metrics_gru['rmse'], metrics_gru['mape']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        bars1 = ax.bar(x - width/2, lstm_values, width, label='LSTM', 
                      color='blue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, gru_values, width, label='GRU', 
                      color='green', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Metric', fontsize=13)
        ax.set_ylabel('Value', fontsize=13)
        ax.set_title('LSTM vs GRU Prediction Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Metrics comparison saved to {filepath}")
        return filepath
    
    def plot_link_congestion_heatmap(self, congestion_matrix, link_labels=None,
                                    output_filename='link_congestion_heatmap.png',
                                    figsize=(14, 10), dpi=300):
        """
        Heatmap showing congestion levels across different links over time.
        
        Args:
            congestion_matrix: (num_links, timesteps) array
            link_labels: List of link names
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        im = ax.imshow(congestion_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        ax.set_xlabel('Time Step', fontsize=13)
        ax.set_ylabel('Link ID', fontsize=13)
        ax.set_title('Inter-Satellite Link Congestion Over Time', fontsize=14, fontweight='bold')
        
        if link_labels and len(link_labels) <= 20:
            ax.set_yticks(np.arange(len(link_labels)))
            ax.set_yticklabels(link_labels, fontsize=8)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Congestion Level (%)', fontsize=12)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Congestion heatmap saved to {filepath}")
        return filepath
    
    def plot_prediction_accuracy_over_time(self, mae_history, rmse_history,
                                          output_filename='accuracy_over_time.png',
                                          figsize=(12, 5), dpi=300):
        """Plot how prediction accuracy changes over the simulation."""
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        timesteps = np.arange(len(mae_history))
        
        # MAE over time
        axes[0].plot(timesteps, mae_history, color='blue', linewidth=2)
        axes[0].set_xlabel('Evaluation Window', fontsize=12)
        axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[0].set_title('Prediction MAE Over Time', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # RMSE over time
        axes[1].plot(timesteps, rmse_history, color='green', linewidth=2)
        axes[1].set_xlabel('Evaluation Window', fontsize=12)
        axes[1].set_ylabel('Root Mean Squared Error', fontsize=12)
        axes[1].set_title('Prediction RMSE Over Time', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Accuracy over time saved to {filepath}")
        return filepath
