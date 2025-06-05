#!/usr/bin/env python3
"""
CLIP-Fields Thor Integration Example
===================================

This script demonstrates how to use the CLIP-Fields integration with AI2-THOR
for semantic navigation tasks. It shows the complete workflow from starting
the semantic memory server to executing navigation tasks.
"""

import sys
import time
import logging
import subprocess
import signal
import os
from pathlib import Path
from typing import List, Dict, Any

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "bridge"))
sys.path.append(str(project_root / "thor_env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationDemo:
    """Demonstrates the CLIP-Fields Thor integration."""
    
    def __init__(self):
        self.server_process = None
        self.project_root = project_root
        
    def start_clipfields_server(self):
        """Start the CLIP-Fields server in the background."""
        logger.info("Starting CLIP-Fields server...")
        
        # Path to server script
        server_script = self.project_root / "clipfields_env" / "server.py"
        
        # Start server process
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root / "bridge")
        
        self.server_process = subprocess.Popen(
            ["python", str(server_script)],
            cwd=str(self.project_root / "clipfields_env"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        time.sleep(5)
        
        # Check if server is running
        if self.server_process.poll() is None:
            logger.info("CLIP-Fields server started successfully")
        else:
            logger.error("Failed to start CLIP-Fields server")
            stdout, stderr = self.server_process.communicate()
            logger.error(f"Server stdout: {stdout.decode()}")
            logger.error(f"Server stderr: {stderr.decode()}")
            raise RuntimeError("Server startup failed")
    
    def stop_clipfields_server(self):
        """Stop the CLIP-Fields server."""
        if self.server_process:
            logger.info("Stopping CLIP-Fields server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            logger.info("CLIP-Fields server stopped")
    
    def run_basic_demo(self):
        """Run a basic demonstration of the integration."""
        logger.info("Running basic integration demo...")
        
        try:
            # Import AI2-THOR integration components
            from thor_integration import (
                Controller, CLIPFieldsClient, NavigationTask, 
                TaskExecutor, create_sample_tasks
            )
            
            # Initialize AI2-THOR controller
            logger.info("Initializing AI2-THOR controller...")
            controller = Controller(
                agentMode="default",
                visibilityDistance=1.5,
                scene="FloorPlan1",
                gridSize=0.25,
                width=224,
                height=224
            )
            
            # Initialize semantic client
            logger.info("Connecting to CLIP-Fields server...")
            semantic_client = CLIPFieldsClient()
            
            # Test connection
            status = semantic_client.get_field_status()
            logger.info(f"Server status: {status}")
            
            # Create task executor
            executor = TaskExecutor(controller, semantic_client)
            
            # Create sample tasks
            tasks = [
                NavigationTask(
                    task_type='ObjectNav',
                    target_description='apple on the counter',
                    scene_name='FloorPlan1',
                    max_steps=100  # Reduced for demo
                ),
                NavigationTask(
                    task_type='ObjectNav', 
                    target_description='mug on the table',
                    scene_name='FloorPlan1',
                    max_steps=100
                )
            ]
            
            logger.info(f"Running {len(tasks)} navigation tasks...")
            
            # Run evaluation
            results = executor.run_evaluation(tasks)
            
            # Display results
            self.display_results(results)
            
            # Cleanup
            controller.stop()
            semantic_client.disconnect()
            
            logger.info("Basic demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    def run_semantic_query_demo(self):
        """Demonstrate semantic querying capabilities."""
        logger.info("Running semantic query demo...")
        
        try:
            from communication_bridge import CLIPFieldsClient, SemanticQuery
            
            # Connect to server
            client = CLIPFieldsClient()
            
            # Reset field with demo bounds
            spatial_bounds = (-5.0, 5.0, -5.0, 5.0, 0.0, 3.0)
            client.reset_field(spatial_bounds)
            
            # Test queries
            test_queries = [
                "red apple on the counter",
                "ceramic mug on the table", 
                "blue book on the shelf",
                "wooden chair near the window"
            ]
            
            logger.info("Testing semantic queries...")
            for query_text in test_queries:
                query = SemanticQuery(
                    text=query_text,
                    resolution=0.1,
                    max_points=500
                )
                
                result = client.query_semantic_field(query)
                
                logger.info(f"Query: '{query_text}'")
                logger.info(f"  Max probability location: {result.max_prob_location}")
                logger.info(f"  Confidence: {result.confidence:.3f}")
                logger.info(f"  Processing time: {result.processing_time:.1f}ms")
                logger.info("")
            
            # Get performance stats
            stats = client.get_performance_stats()
            logger.info(f"Performance statistics: {stats}")
            
            client.disconnect()
            logger.info("Semantic query demo completed!")
            
        except Exception as e:
            logger.error(f"Semantic query demo failed: {e}")
            raise
    
    def display_results(self, results: List[Any]):
        """Display evaluation results in a formatted way."""
        logger.info("\\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        
        for i, result in enumerate(results):
            logger.info(f"\\nTask {i+1}: {result.metadata['target_description']}")
            logger.info(f"  Task Type: {result.metadata['task_type']}")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Steps Taken: {result.steps_taken}")
            logger.info(f"  Path Length: {result.path_length:.2f}m")
            logger.info(f"  SPL: {result.spl:.3f}")
            logger.info(f"  Goal Distance: {result.goal_distance:.2f}m")
            logger.info(f"  Execution Time: {result.execution_time:.1f}s")
            logger.info(f"  Semantic Queries: {result.semantic_queries}")
        
        # Summary statistics
        if results:
            success_rate = sum(r.success for r in results) / len(results)
            avg_spl = sum(r.spl for r in results) / len(results)
            avg_steps = sum(r.steps_taken for r in results) / len(results)
            avg_queries = sum(r.semantic_queries for r in results) / len(results)
            
            logger.info("\\n" + "-"*40)
            logger.info("SUMMARY STATISTICS")
            logger.info("-"*40)
            logger.info(f"Success Rate: {success_rate:.3f}")
            logger.info(f"Average SPL: {avg_spl:.3f}")
            logger.info(f"Average Steps: {avg_steps:.1f}")
            logger.info(f"Average Semantic Queries: {avg_queries:.1f}")
        
        logger.info("="*60)
    
    def run_full_demo(self):
        """Run the complete demonstration."""
        try:
            # Start server
            self.start_clipfields_server()
            
            # Run demos
            self.run_semantic_query_demo()
            self.run_basic_demo()
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            # Always stop server
            self.stop_clipfields_server()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP-Fields Thor Integration Demo")
    parser.add_argument(
        "--demo-type", 
        choices=["full", "query", "navigation"],
        default="full",
        help="Type of demo to run"
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Don't start/stop server (assume it's already running)"
    )
    
    args = parser.parse_args()
    
    demo = IntegrationDemo()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, shutting down...")
        demo.stop_clipfields_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.demo_type == "full":
            demo.run_full_demo()
        elif args.demo_type == "query":
            if not args.no_server:
                demo.start_clipfields_server()
            demo.run_semantic_query_demo()
            if not args.no_server:
                demo.stop_clipfields_server()
        elif args.demo_type == "navigation":
            if not args.no_server:
                demo.start_clipfields_server()
            demo.run_basic_demo()
            if not args.no_server:
                demo.stop_clipfields_server()
        
        logger.info("Demo completed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

