#!/usr/bin/env python3
"""
Quick Evaluation Script
======================

This script provides quick evaluation of the current model performance
on a small set of tasks for rapid feedback during training.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "bridge"))
sys.path.append(str(project_root / "thor_env"))

from ai2thor.controller import Controller
from communication_bridge import CLIPFieldsClient
from thor_integration import NavigationTask, TaskExecutor, TaskResult

class QuickEvaluator:
    """Quick evaluation for training feedback."""
    
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path
        self.controller = None
        self.client = None
        self.executor = None
        
    def initialize_components(self, scene: str = "FloorPlan1"):
        """Initialize evaluation components."""
        print(f"Initializing evaluation components for scene: {scene}")
        
        # Initialize AI2-THOR controller
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene=scene,
            gridSize=0.25,
            width=224,
            height=224
        )
        
        # Initialize semantic client
        self.client = CLIPFieldsClient()
        
        # Test connection
        try:
            status = self.client.get_field_status()
            print(f"Connected to CLIP-Fields server")
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            raise
        
        # Initialize task executor
        self.executor = TaskExecutor(self.controller, self.client)
        
        # Reset field for scene
        spatial_bounds = self._get_scene_bounds(scene)
        self.client.reset_field(spatial_bounds)
        
    def _get_scene_bounds(self, scene_name: str) -> tuple:
        """Get spatial bounds for scene."""
        scene_bounds = {
            'FloorPlan1': (-6.0, 6.0, -6.0, 6.0, 0.0, 3.0),
            'FloorPlan2': (-8.0, 8.0, -8.0, 8.0, 0.0, 3.0),
            'FloorPlan5': (-7.0, 7.0, -7.0, 7.0, 0.0, 3.0),
        }
        return scene_bounds.get(scene_name, (-10.0, 10.0, -10.0, 10.0, 0.0, 4.0))
    
    def create_quick_eval_tasks(self, num_tasks: int = 10, scene: str = "FloorPlan1") -> List[NavigationTask]:
        """Create a small set of evaluation tasks."""
        # Quick evaluation targets (mix of easy and medium difficulty)
        eval_targets = [
            "dining table with chairs",
            "comfortable sofa for sitting",
            "kitchen counter with appliances",
            "bed in the bedroom",
            "television on the stand",
            "refrigerator in the kitchen",
            "office desk with computer",
            "bookshelf with books",
            "bathroom sink with mirror",
            "coffee table in living room"
        ]
        
        tasks = []
        for i in range(num_tasks):
            target = eval_targets[i % len(eval_targets)]
            
            task = NavigationTask(
                task_type='ObjectNav',
                target_description=target,
                scene_name=scene,
                max_steps=300,  # Shorter for quick eval
                success_distance=1.0
            )
            tasks.append(task)
        
        return tasks
    
    def run_quick_evaluation(self, num_tasks: int = 10, scene: str = "FloorPlan1") -> Dict[str, Any]:
        """Run quick evaluation and return results."""
        print(f"Running quick evaluation with {num_tasks} tasks on {scene}")
        
        # Initialize components
        self.initialize_components(scene)
        
        # Create evaluation tasks
        tasks = self.create_quick_eval_tasks(num_tasks, scene)
        
        # Run evaluation
        results = []
        start_time = time.time()
        
        for i, task in enumerate(tasks):
            print(f"Task {i+1}/{num_tasks}: {task.target_description}")
            
            # Reset environment
            self.controller.reset()
            
            # Execute task
            result = self.executor.agent.execute_task(task)
            results.append(result)
            
            print(f"  Result: Success={result.success}, Steps={result.steps_taken}, SPL={result.spl:.3f}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, total_time)
        
        # Print summary
        self._print_summary(metrics, scene, num_tasks)
        
        return metrics
    
    def _calculate_metrics(self, results: List[TaskResult], total_time: float) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        if not results:
            return {}
        
        # Basic metrics
        success_rate = sum(r.success for r in results) / len(results)
        avg_spl = np.mean([r.spl for r in results])
        avg_steps = np.mean([r.steps_taken for r in results])
        avg_path_length = np.mean([r.path_length for r in results])
        avg_execution_time = np.mean([r.execution_time for r in results])
        avg_semantic_queries = np.mean([r.semantic_queries for r in results])
        
        # Success-specific metrics
        successful_results = [r for r in results if r.success]
        if successful_results:
            success_avg_spl = np.mean([r.spl for r in successful_results])
            success_avg_steps = np.mean([r.steps_taken for r in successful_results])
        else:
            success_avg_spl = 0.0
            success_avg_steps = 0.0
        
        # Failure analysis
        failed_results = [r for r in results if not r.success]
        if failed_results:
            failure_avg_steps = np.mean([r.steps_taken for r in failed_results])
            failure_avg_queries = np.mean([r.semantic_queries for r in failed_results])
        else:
            failure_avg_steps = 0.0
            failure_avg_queries = 0.0
        
        return {
            'num_tasks': len(results),
            'success_rate': success_rate,
            'avg_spl': avg_spl,
            'avg_steps': avg_steps,
            'avg_path_length': avg_path_length,
            'avg_execution_time': avg_execution_time,
            'avg_semantic_queries': avg_semantic_queries,
            'total_evaluation_time': total_time,
            'success_metrics': {
                'count': len(successful_results),
                'avg_spl': success_avg_spl,
                'avg_steps': success_avg_steps
            },
            'failure_metrics': {
                'count': len(failed_results),
                'avg_steps': failure_avg_steps,
                'avg_queries': failure_avg_queries
            },
            'detailed_results': [
                {
                    'target': r.metadata.get('target_description', 'unknown'),
                    'success': r.success,
                    'steps': r.steps_taken,
                    'spl': r.spl,
                    'queries': r.semantic_queries
                }
                for r in results
            ]
        }
    
    def _print_summary(self, metrics: Dict[str, Any], scene: str, num_tasks: int):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("QUICK EVALUATION RESULTS")
        print("="*60)
        print(f"Scene: {scene}")
        print(f"Tasks: {num_tasks}")
        print(f"Evaluation Time: {metrics['total_evaluation_time']:.1f} seconds")
        print()
        
        print("OVERALL PERFORMANCE:")
        print(f"  Success Rate: {metrics['success_rate']:.1%}")
        print(f"  Average SPL: {metrics['avg_spl']:.3f}")
        print(f"  Average Steps: {metrics['avg_steps']:.1f}")
        print(f"  Average Path Length: {metrics['avg_path_length']:.2f}m")
        print(f"  Average Execution Time: {metrics['avg_execution_time']:.1f}s")
        print(f"  Average Semantic Queries: {metrics['avg_semantic_queries']:.1f}")
        print()
        
        if metrics['success_metrics']['count'] > 0:
            print("SUCCESSFUL TASKS:")
            print(f"  Count: {metrics['success_metrics']['count']}")
            print(f"  Average SPL: {metrics['success_metrics']['avg_spl']:.3f}")
            print(f"  Average Steps: {metrics['success_metrics']['avg_steps']:.1f}")
            print()
        
        if metrics['failure_metrics']['count'] > 0:
            print("FAILED TASKS:")
            print(f"  Count: {metrics['failure_metrics']['count']}")
            print(f"  Average Steps: {metrics['failure_metrics']['avg_steps']:.1f}")
            print(f"  Average Queries: {metrics['failure_metrics']['avg_queries']:.1f}")
            print()
        
        # Performance assessment
        if metrics['success_rate'] >= 0.8:
            assessment = "EXCELLENT"
        elif metrics['success_rate'] >= 0.6:
            assessment = "GOOD"
        elif metrics['success_rate'] >= 0.4:
            assessment = "FAIR"
        else:
            assessment = "NEEDS IMPROVEMENT"
        
        print(f"PERFORMANCE ASSESSMENT: {assessment}")
        print("="*60)
    
    def save_results(self, metrics: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp and metadata
        results_data = {
            'timestamp': time.time(),
            'evaluation_type': 'quick_evaluation',
            'checkpoint_path': self.checkpoint_path,
            'metrics': metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.controller:
            self.controller.stop()
        if self.client:
            self.client.disconnect()


def main():
    """Main function for quick evaluation."""
    parser = argparse.ArgumentParser(description="Quick Evaluation Script")
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--tasks', type=int, default=10, help='Number of evaluation tasks')
    parser.add_argument('--scene', type=str, default='FloorPlan1', help='AI2-THOR scene')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = QuickEvaluator(args.checkpoint)
    
    try:
        # Run evaluation
        metrics = evaluator.run_quick_evaluation(args.tasks, args.scene)
        
        # Save results if requested
        if args.output:
            evaluator.save_results(metrics, args.output)
        
        # Print detailed results if verbose
        if args.verbose:
            print("\nDETAILED RESULTS:")
            for i, result in enumerate(metrics['detailed_results']):
                print(f"Task {i+1}: {result['target']}")
                print(f"  Success: {result['success']}")
                print(f"  Steps: {result['steps']}")
                print(f"  SPL: {result['spl']:.3f}")
                print(f"  Queries: {result['queries']}")
                print()
        
        # Return appropriate exit code
        if metrics['success_rate'] >= 0.5:
            print("✓ Evaluation passed (≥50% success rate)")
            exit_code = 0
        else:
            print("✗ Evaluation failed (<50% success rate)")
            exit_code = 1
        
        return exit_code
        
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

