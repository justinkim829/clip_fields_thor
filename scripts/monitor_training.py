#!/usr/bin/env python3
"""
Training Progress Monitor
========================

This script monitors training progress in real-time, displaying metrics
and visualizations of the learning process.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "bridge"))

# Configure matplotlib for non-interactive use
plt.switch_backend('Agg')

class TrainingMonitor:
    """Monitors and visualizes training progress."""
    
    def __init__(self, checkpoint_dir: str, refresh_interval: int = 30):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.refresh_interval = refresh_interval
        self.metrics_history = []
        self.last_update = 0
        
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the latest training checkpoint."""
        latest_file = self.checkpoint_dir / "latest.json"
        
        if not latest_file.exists():
            return None
        
        # Check if file has been updated
        file_mtime = latest_file.stat().st_mtime
        if file_mtime <= self.last_update:
            return None
        
        self.last_update = file_mtime
        
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def extract_metrics(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from checkpoint."""
        training_stats = checkpoint.get('training_stats', {})
        training_state = checkpoint.get('training_state', {})
        
        # Calculate derived metrics
        total_tasks = training_stats.get('successful_tasks', 0) + training_stats.get('failed_tasks', 0)
        success_rate = training_stats.get('successful_tasks', 0) / max(total_tasks, 1)
        
        # Calculate training speed (episodes per hour)
        start_time = training_state.get('start_time')
        current_episode = training_state.get('current_episode', 0)
        
        if start_time:
            elapsed_hours = (time.time() - start_time) / 3600
            episodes_per_hour = current_episode / max(elapsed_hours, 0.01)
        else:
            episodes_per_hour = 0
        
        return {
            'timestamp': time.time(),
            'current_episode': current_episode,
            'current_stage': training_state.get('current_stage', 'unknown'),
            'success_rate': success_rate,
            'average_spl': training_stats.get('average_spl', 0),
            'total_observations': training_stats.get('total_observations', 0),
            'total_queries': training_stats.get('total_queries', 0),
            'episodes_per_hour': episodes_per_hour,
            'best_performance': training_state.get('best_performance', 0),
            'scene_performance': training_stats.get('scene_performance', {}),
            'stage_performance': training_stats.get('stage_performance', {})
        }
    
    def update_metrics_history(self, metrics: Dict[str, Any]):
        """Update the metrics history."""
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 data points
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def print_current_status(self, metrics: Dict[str, Any]):
        """Print current training status to console."""
        print("\n" + "="*60)
        print("CLIP-FIELDS TRAINING MONITOR")
        print("="*60)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Stage: {metrics['current_stage'].upper()}")
        print(f"Episode: {metrics['current_episode']}")
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        print(f"Average SPL: {metrics['average_spl']:.3f}")
        print(f"Best Performance: {metrics['best_performance']:.1%}")
        print(f"Training Speed: {metrics['episodes_per_hour']:.1f} episodes/hour")
        print(f"Total Observations: {metrics['total_observations']:,}")
        print(f"Total Queries: {metrics['total_queries']:,}")
        
        # Scene performance breakdown
        if metrics['scene_performance']:
            print("\nScene Performance:")
            for scene, perf in metrics['scene_performance'].items():
                if perf['episodes'] > 0:
                    scene_success = perf['successes'] / perf['episodes']
                    scene_spl = np.mean(perf['spl_scores']) if perf['spl_scores'] else 0
                    print(f"  {scene}: {scene_success:.1%} success, {scene_spl:.3f} SPL ({perf['episodes']} episodes)")
        
        print("="*60)
    
    def generate_plots(self, output_dir: str = None):
        """Generate training progress plots."""
        if len(self.metrics_history) < 2:
            return
        
        if output_dir is None:
            output_dir = self.checkpoint_dir / "plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Extract time series data
        timestamps = [m['timestamp'] for m in self.metrics_history]
        episodes = [m['current_episode'] for m in self.metrics_history]
        success_rates = [m['success_rate'] for m in self.metrics_history]
        spl_scores = [m['average_spl'] for m in self.metrics_history]
        observations = [m['total_observations'] for m in self.metrics_history]
        
        # Convert timestamps to hours from start
        start_time = timestamps[0]
        hours = [(t - start_time) / 3600 for t in timestamps]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Success Rate over Time
        ax1.plot(hours, success_rates, 'b-', linewidth=2)
        ax1.set_xlabel('Training Time (hours)')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate Progress')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: SPL over Time
        ax2.plot(hours, spl_scores, 'g-', linewidth=2)
        ax2.set_xlabel('Training Time (hours)')
        ax2.set_ylabel('Average SPL')
        ax2.set_title('SPL Progress')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Episodes over Time
        ax3.plot(hours, episodes, 'r-', linewidth=2)
        ax3.set_xlabel('Training Time (hours)')
        ax3.set_ylabel('Episodes Completed')
        ax3.set_title('Training Progress')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Observations over Time
        ax4.plot(hours, observations, 'm-', linewidth=2)
        ax4.set_xlabel('Training Time (hours)')
        ax4.set_ylabel('Total Observations')
        ax4.set_title('Data Collection Progress')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"training_progress_{int(time.time())}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {plot_path}")
    
    def generate_stage_analysis(self, output_dir: str = None):
        """Generate stage-by-stage performance analysis."""
        if not self.metrics_history:
            return
        
        if output_dir is None:
            output_dir = self.checkpoint_dir / "plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Get latest stage performance data
        latest_metrics = self.metrics_history[-1]
        stage_performance = latest_metrics.get('stage_performance', {})
        
        if not stage_performance:
            return
        
        # Create stage comparison plot
        stages = list(stage_performance.keys())
        if not stages:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Success Rate by Stage
        stage_success_rates = []
        stage_spls = []
        
        for stage in stages:
            stage_data = stage_performance[stage]
            if stage_data:
                latest_perf = stage_data[-1]  # Get latest performance for this stage
                stage_success_rates.append(latest_perf['success_rate'])
                stage_spls.append(latest_perf['avg_spl'])
            else:
                stage_success_rates.append(0)
                stage_spls.append(0)
        
        x_pos = np.arange(len(stages))
        
        ax1.bar(x_pos, stage_success_rates, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Training Stage')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Training Stage')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stages, rotation=45)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(stage_success_rates):
            ax1.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # Plot 2: SPL by Stage
        ax2.bar(x_pos, stage_spls, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Training Stage')
        ax2.set_ylabel('Average SPL')
        ax2.set_title('SPL by Training Stage')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stages, rotation=45)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(stage_spls):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"stage_analysis_{int(time.time())}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Stage analysis saved to: {plot_path}")
    
    def run_monitoring(self, generate_plots: bool = True):
        """Run continuous monitoring loop."""
        print(f"Starting training monitor...")
        print(f"Monitoring directory: {self.checkpoint_dir}")
        print(f"Refresh interval: {self.refresh_interval} seconds")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Load latest checkpoint
                checkpoint = self.load_latest_checkpoint()
                
                if checkpoint:
                    # Extract and update metrics
                    metrics = self.extract_metrics(checkpoint)
                    self.update_metrics_history(metrics)
                    
                    # Print status
                    self.print_current_status(metrics)
                    
                    # Generate plots if requested
                    if generate_plots and len(self.metrics_history) >= 2:
                        self.generate_plots()
                        self.generate_stage_analysis()
                
                else:
                    print(f"No checkpoint updates found at {time.strftime('%H:%M:%S')}")
                
                # Wait for next update
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Monitoring error: {e}")
    
    def generate_summary_report(self, output_path: str = None):
        """Generate a comprehensive training summary report."""
        if not self.metrics_history:
            print("No training data available for report generation")
            return
        
        if output_path is None:
            output_path = self.checkpoint_dir / "training_summary.txt"
        else:
            output_path = Path(output_path)
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate training duration
        if len(self.metrics_history) > 1:
            start_time = self.metrics_history[0]['timestamp']
            end_time = self.metrics_history[-1]['timestamp']
            duration_hours = (end_time - start_time) / 3600
        else:
            duration_hours = 0
        
        # Generate report
        report = []
        report.append("CLIP-FIELDS TRAINING SUMMARY REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Training Duration: {duration_hours:.1f} hours")
        report.append("")
        
        report.append("FINAL PERFORMANCE:")
        report.append(f"  Success Rate: {latest_metrics['success_rate']:.1%}")
        report.append(f"  Average SPL: {latest_metrics['average_spl']:.3f}")
        report.append(f"  Best Performance: {latest_metrics['best_performance']:.1%}")
        report.append(f"  Episodes Completed: {latest_metrics['current_episode']}")
        report.append(f"  Total Observations: {latest_metrics['total_observations']:,}")
        report.append(f"  Total Queries: {latest_metrics['total_queries']:,}")
        report.append("")
        
        # Scene performance
        scene_perf = latest_metrics.get('scene_performance', {})
        if scene_perf:
            report.append("SCENE PERFORMANCE:")
            for scene, perf in scene_perf.items():
                if perf['episodes'] > 0:
                    scene_success = perf['successes'] / perf['episodes']
                    scene_spl = np.mean(perf['spl_scores']) if perf['spl_scores'] else 0
                    report.append(f"  {scene}:")
                    report.append(f"    Success Rate: {scene_success:.1%}")
                    report.append(f"    Average SPL: {scene_spl:.3f}")
                    report.append(f"    Episodes: {perf['episodes']}")
            report.append("")
        
        # Training efficiency
        if duration_hours > 0:
            episodes_per_hour = latest_metrics['current_episode'] / duration_hours
            observations_per_hour = latest_metrics['total_observations'] / duration_hours
            
            report.append("TRAINING EFFICIENCY:")
            report.append(f"  Episodes per Hour: {episodes_per_hour:.1f}")
            report.append(f"  Observations per Hour: {observations_per_hour:,.0f}")
            report.append("")
        
        # Performance trends
        if len(self.metrics_history) >= 10:
            recent_success = np.mean([m['success_rate'] for m in self.metrics_history[-10:]])
            early_success = np.mean([m['success_rate'] for m in self.metrics_history[:10]])
            improvement = recent_success - early_success
            
            report.append("LEARNING PROGRESS:")
            report.append(f"  Early Success Rate: {early_success:.1%}")
            report.append(f"  Recent Success Rate: {recent_success:.1%}")
            report.append(f"  Improvement: {improvement:+.1%}")
            report.append("")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Training summary report saved to: {output_path}")
        
        # Also print to console
        print("\n" + '\n'.join(report))


def main():
    """Main function for training monitor."""
    parser = argparse.ArgumentParser(description="Training Progress Monitor")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/',
                       help='Directory containing training checkpoints')
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval in seconds')
    parser.add_argument('--plots', action='store_true',
                       help='Generate progress plots')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate summary report and exit')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = TrainingMonitor(args.checkpoint_dir, args.refresh)
    
    if args.report_only:
        # Load all available data and generate report
        checkpoint = monitor.load_latest_checkpoint()
        if checkpoint:
            metrics = monitor.extract_metrics(checkpoint)
            monitor.update_metrics_history(metrics)
            
            output_path = None
            if args.output_dir:
                output_path = Path(args.output_dir) / "training_summary.txt"
            
            monitor.generate_summary_report(output_path)
        else:
            print("No checkpoint data found")
    else:
        # Run continuous monitoring
        monitor.run_monitoring(args.plots)


if __name__ == "__main__":
    main()

