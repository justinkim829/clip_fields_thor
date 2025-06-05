#!/usr/bin/env python3
"""
Rapid Training Script for Demo-Ready Performance
===============================================

This script implements rapid training for achieving demo-ready performance
in 2-4 hours. It's designed to work with the existing CLIP-Fields Thor
integration architecture.
"""

import sys
import time
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "bridge"))
sys.path.append(str(project_root / "thor_env"))

from ai2thor.controller import Controller
from communication_bridge import CLIPFieldsClient, Observation, SemanticQuery
from thor_integration import (
    NavigationTask, TaskExecutor, SemanticNavigationAgent,
    ObservationManager, create_sample_tasks
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RapidTrainer:
    """Implements rapid training for demo-ready performance."""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.controller = None
        self.client = None
        self.agent = None
        self.observation_manager = None

        # Training statistics
        self.training_stats = {
            'episodes_completed': 0,
            'total_observations': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'start_time': None
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default rapid training configuration
            config = {
                'clip_fields': {
                    'batch_size': 16,
                    'num_levels': 8,
                    'resolution': 0.1,
                    'max_episodes': 100,
                    'update_frequency': 5,
                    'learning_rate': 0.005
                },
                'thor': {
                    'max_steps_per_task': 200,
                    'scenes': ['FloorPlan1'],
                    'success_distance': 1.5,
                    'grid_size': 0.25
                },
                'training': {
                    'exploration_episodes': 50,
                    'object_learning_episodes': 30,
                    'validation_episodes': 20,
                    'checkpoint_frequency': 25
                }
            }
        return config

    def initialize_components(self):
        """Initialize AI2-THOR and CLIP-Fields components."""
        logger.info("Initializing training components...")

        # Initialize AI2-THOR controller
        thor_config = self.config['thor']
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene=thor_config['scenes'][0],
            gridSize=thor_config['grid_size'],
            commit_id="v2.1.0",
            width=224,
            height=224
        )

        # Initialize semantic client
        self.client = CLIPFieldsClient()

        # Test connection
        try:
            status = self.client.get_field_status()
            logger.info(f"Connected to CLIP-Fields server: {status}")
        except Exception as e:
            logger.error(f"Failed to connect to CLIP-Fields server: {e}")
            raise

        # Initialize observation manager and agent
        self.observation_manager = ObservationManager(self.controller)
        self.agent = SemanticNavigationAgent(self.controller, self.client)

        # Reset field with scene bounds
        spatial_bounds = self._get_scene_bounds(thor_config['scenes'][0])
        self.client.reset_field(spatial_bounds)

        logger.info("Components initialized successfully")

    def _get_scene_bounds(self, scene_name: str) -> tuple:
        """Get spatial bounds for a scene."""
        # Scene-specific bounds (these could be loaded from metadata)
        scene_bounds = {
            'FloorPlan1': (-6.0, 6.0, -6.0, 6.0, 0.0, 3.0),
            'FloorPlan2': (-8.0, 8.0, -8.0, 8.0, 0.0, 3.0),
            'FloorPlan5': (-7.0, 7.0, -7.0, 7.0, 0.0, 3.0),
        }
        return scene_bounds.get(scene_name, (-10.0, 10.0, -10.0, 10.0, 0.0, 4.0))

    def run_exploration_phase(self, episodes: int):
        """Run exploration phase to build basic spatial understanding."""
        logger.info(f"Starting exploration phase: {episodes} episodes")
        self.training_stats['start_time'] = time.time()

        for episode in range(episodes):
            logger.info(f"Exploration episode {episode + 1}/{episodes}")

            # Reset agent position
            self.controller.reset()

            # Random exploration
            steps = 0
            max_steps = self.config['thor']['max_steps_per_task']

            while steps < max_steps:
                # Capture observation
                observation = self.observation_manager.capture_observation()

                # Push observation to semantic field
                success = self.client.push_observation(observation)
                if success:
                    self.training_stats['total_observations'] += 1

                # Random action for exploration
                action = np.random.choice([
                    'MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight'
                ])

                event = self.controller.step(action=action)
                if not event.metadata['lastActionSuccess']:
                    # Try a different action if stuck
                    action = np.random.choice(['RotateLeft', 'RotateRight'])
                    self.controller.step(action=action)

                steps += 1

                # Update frequency check
                if steps % self.config['clip_fields']['update_frequency'] == 0:
                    self._log_progress(episode, episodes, steps, max_steps)

            self.training_stats['episodes_completed'] += 1

            # Checkpoint if needed
            if (episode + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self._save_checkpoint(f"exploration_ep_{episode + 1}")

        logger.info("Exploration phase completed")

    def run_object_learning_phase(self, episodes: int):
        """Run object learning phase with semantic queries."""
        logger.info(f"Starting object learning phase: {episodes} episodes")

        # Common objects to learn
        target_objects = [
            "table in the room",
            "chair near the table",
            "sofa in the living room",
            "bed in the bedroom",
            "refrigerator in the kitchen",
            "microwave on the counter",
            "television on the stand",
            "lamp on the table"
        ]

        for episode in range(episodes):
            logger.info(f"Object learning episode {episode + 1}/{episodes}")

            # Reset agent position
            self.controller.reset()

            # Select target object for this episode
            target = target_objects[episode % len(target_objects)]

            # Create navigation task
            task = NavigationTask(
                task_type='ObjectNav',
                target_description=target,
                scene_name=self.config['thor']['scenes'][0],
                max_steps=self.config['thor']['max_steps_per_task'],
                success_distance=self.config['thor']['success_distance']
            )

            # Execute task (this will generate queries and updates)
            result = self.agent.execute_task(task)

            # Update statistics
            self.training_stats['episodes_completed'] += 1
            if result.success:
                self.training_stats['successful_queries'] += result.semantic_queries
            else:
                self.training_stats['failed_queries'] += result.semantic_queries

            logger.info(f"Task result: Success={result.success}, "
                       f"Steps={result.steps_taken}, "
                       f"Queries={result.semantic_queries}")

            # Checkpoint if needed
            if (episode + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self._save_checkpoint(f"object_learning_ep_{episode + 1}")

        logger.info("Object learning phase completed")

    def run_validation_phase(self, episodes: int):
        """Run validation to test current performance."""
        logger.info(f"Starting validation phase: {episodes} episodes")

        # Create diverse validation tasks
        validation_tasks = [
            NavigationTask(
                task_type='ObjectNav',
                target_description='dining table with chairs',
                scene_name=self.config['thor']['scenes'][0],
                max_steps=self.config['thor']['max_steps_per_task']
            ),
            NavigationTask(
                task_type='ObjectNav',
                target_description='comfortable sofa for sitting',
                scene_name=self.config['thor']['scenes'][0],
                max_steps=self.config['thor']['max_steps_per_task']
            ),
            NavigationTask(
                task_type='ObjectNav',
                target_description='kitchen counter with appliances',
                scene_name=self.config['thor']['scenes'][0],
                max_steps=self.config['thor']['max_steps_per_task']
            )
        ]

        results = []
        for episode in range(episodes):
            task = validation_tasks[episode % len(validation_tasks)]
            logger.info(f"Validation {episode + 1}/{episodes}: {task.target_description}")

            # Reset for validation
            self.controller.reset()

            # Execute validation task
            result = self.agent.execute_task(task)
            results.append(result)

            logger.info(f"Validation result: Success={result.success}, "
                       f"SPL={result.spl:.3f}")

        # Calculate validation metrics
        success_rate = sum(r.success for r in results) / len(results)
        avg_spl = np.mean([r.spl for r in results])
        avg_steps = np.mean([r.steps_taken for r in results])

        logger.info(f"Validation Results:")
        logger.info(f"  Success Rate: {success_rate:.3f}")
        logger.info(f"  Average SPL: {avg_spl:.3f}")
        logger.info(f"  Average Steps: {avg_steps:.1f}")

        return {
            'success_rate': success_rate,
            'avg_spl': avg_spl,
            'avg_steps': avg_steps,
            'results': results
        }

    def _log_progress(self, episode: int, total_episodes: int,
                     step: int, total_steps: int):
        """Log training progress."""
        elapsed = time.time() - self.training_stats['start_time']
        logger.info(f"Episode {episode + 1}/{total_episodes}, "
                   f"Step {step}/{total_steps}, "
                   f"Elapsed: {elapsed/3600:.1f}h, "
                   f"Observations: {self.training_stats['total_observations']}")

    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = project_root / "checkpoints" / "rapid_training"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.json"

        checkpoint_data = {
            'training_stats': self.training_stats,
            'config': self.config,
            'timestamp': time.time()
        }

        with open(checkpoint_path, 'w') as f:
            import json
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def run_full_training(self):
        """Run complete rapid training pipeline."""
        logger.info("Starting rapid training for demo-ready performance")

        try:
            # Initialize components
            self.initialize_components()

            # Phase 1: Exploration
            self.run_exploration_phase(
                self.config['training']['exploration_episodes']
            )

            # Phase 2: Object Learning
            self.run_object_learning_phase(
                self.config['training']['object_learning_episodes']
            )

            # Phase 3: Validation
            validation_results = self.run_validation_phase(
                self.config['training']['validation_episodes']
            )

            # Final statistics
            total_time = time.time() - self.training_stats['start_time']
            logger.info(f"Rapid training completed in {total_time/3600:.1f} hours")
            logger.info(f"Final performance: {validation_results['success_rate']:.1%} success rate")

            # Save final checkpoint
            self._save_checkpoint("final_rapid_training")

            return validation_results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup
            if self.controller:
                self.controller.stop()
            if self.client:
                self.client.disconnect()


def main():
    """Main function for rapid training script."""
    parser = argparse.ArgumentParser(description="Rapid Training for Demo-Ready Performance")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['exploration', 'object_learning', 'validation', 'full'],
                       default='full', help='Training mode')
    parser.add_argument('--episodes', type=int, help='Number of episodes (overrides config)')
    parser.add_argument('--scene', type=str, default='FloorPlan1', help='AI2-THOR scene')

    args = parser.parse_args()

    # Initialize trainer
    trainer = RapidTrainer(args.config)

    # Override config with command line arguments
    if args.episodes:
        if args.mode == 'exploration':
            trainer.config['training']['exploration_episodes'] = args.episodes
        elif args.mode == 'object_learning':
            trainer.config['training']['object_learning_episodes'] = args.episodes
        elif args.mode == 'validation':
            trainer.config['training']['validation_episodes'] = args.episodes

    if args.scene:
        trainer.config['thor']['scenes'] = [args.scene]

    try:
        if args.mode == 'full':
            results = trainer.run_full_training()
        else:
            trainer.initialize_components()

            if args.mode == 'exploration':
                trainer.run_exploration_phase(
                    trainer.config['training']['exploration_episodes']
                )
            elif args.mode == 'object_learning':
                trainer.run_object_learning_phase(
                    trainer.config['training']['object_learning_episodes']
                )
            elif args.mode == 'validation':
                results = trainer.run_validation_phase(
                    trainer.config['training']['validation_episodes']
                )
                print(f"Validation Success Rate: {results['success_rate']:.1%}")

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

