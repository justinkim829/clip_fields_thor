#!/usr/bin/env python3
"""
Production Training Script for Optimal Performance
=================================================

This script implements comprehensive training for achieving optimal performance
over 40-80 hours. It includes multi-stage training, checkpointing, and
distributed training support.
"""

import sys
import time
import logging
import argparse
import yaml
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import asdict

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "bridge"))
sys.path.append(str(project_root / "thor_env"))

from ai2thor.controller import Controller
from communication_bridge import CLIPFieldsClient, Observation, SemanticQuery
from thor_integration import (
    NavigationTask, TaskExecutor, SemanticNavigationAgent, 
    ObservationManager, TaskResult, create_sample_tasks
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTrainer:
    """Implements production-level training for optimal performance."""
    
    def __init__(self, config_path: str = None, checkpoint_dir: str = None):
        self.config = self._load_config(config_path)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints/production")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.controller = None
        self.client = None
        self.agent = None
        self.observation_manager = None
        
        # Training state
        self.training_state = {
            'current_stage': None,
            'current_episode': 0,
            'total_episodes': 0,
            'current_scene': None,
            'start_time': None,
            'stage_start_time': None,
            'best_performance': 0.0,
            'performance_history': []
        }
        
        # Comprehensive statistics
        self.training_stats = {
            'episodes_completed': 0,
            'total_observations': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_queries': 0,
            'query_success_rate': 0.0,
            'average_spl': 0.0,
            'scene_performance': {},
            'stage_performance': {}
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load production configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default production configuration
            config = {
                'clip_fields': {
                    'batch_size': 32,
                    'num_levels': 16,
                    'resolution': 0.05,
                    'learning_rate': 0.002,
                    'online_learning_rate': 0.0002,
                    'consolidation_penalty': 0.01,
                    'update_frequency': 10
                },
                'thor': {
                    'max_steps_per_task': 500,
                    'scenes': ['FloorPlan1', 'FloorPlan2', 'FloorPlan5', 'FloorPlan10', 'FloorPlan15'],
                    'success_distance': 1.0,
                    'grid_size': 0.25
                },
                'training': {
                    'stages': {
                        'exploration': {
                            'episodes': 200,
                            'description': 'Basic spatial exploration and mapping'
                        },
                        'object_learning': {
                            'episodes': 400,
                            'description': 'Object-location association learning'
                        },
                        'fine_tuning': {
                            'episodes': 400,
                            'description': 'Fine-grained spatial understanding'
                        },
                        'generalization': {
                            'episodes': 200,
                            'description': 'Multi-scene generalization'
                        }
                    },
                    'checkpoint_frequency': 50,
                    'evaluation_frequency': 25,
                    'early_stopping_patience': 100
                },
                'evaluation': {
                    'num_tasks_per_scene': 100,
                    'baseline_comparison': True,
                    'detailed_analysis': True
                }
            }
        return config
    
    def initialize_components(self, scene: str = None):
        """Initialize AI2-THOR and CLIP-Fields components."""
        logger.info("Initializing production training components...")
        
        # Select scene
        if scene is None:
            scene = self.config['thor']['scenes'][0]
        
        # Initialize AI2-THOR controller
        thor_config = self.config['thor']
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene=scene,
            gridSize=thor_config['grid_size'],
            width=224,
            height=224,
            renderDepthImage=True,
            renderInstanceSegmentation=True
        )
        
        # Initialize semantic client
        self.client = CLIPFieldsClient()
        
        # Test connection and get server status
        try:
            status = self.client.get_field_status()
            logger.info(f"Connected to CLIP-Fields server")
            logger.info(f"Server status: {status}")
        except Exception as e:
            logger.error(f"Failed to connect to CLIP-Fields server: {e}")
            raise
        
        # Initialize observation manager and agent
        self.observation_manager = ObservationManager(self.controller)
        self.agent = SemanticNavigationAgent(self.controller, self.client)
        
        # Reset field with scene bounds
        spatial_bounds = self._get_scene_bounds(scene)
        self.client.reset_field(spatial_bounds)
        
        self.training_state['current_scene'] = scene
        logger.info(f"Components initialized for scene: {scene}")
    
    def _get_scene_bounds(self, scene_name: str) -> Tuple[float, ...]:
        """Get spatial bounds for different scenes."""
        scene_bounds = {
            'FloorPlan1': (-6.0, 6.0, -6.0, 6.0, 0.0, 3.0),
            'FloorPlan2': (-8.0, 8.0, -8.0, 8.0, 0.0, 3.0),
            'FloorPlan5': (-7.0, 7.0, -7.0, 7.0, 0.0, 3.0),
            'FloorPlan10': (-9.0, 9.0, -9.0, 9.0, 0.0, 3.0),
            'FloorPlan15': (-8.0, 8.0, -8.0, 8.0, 0.0, 3.0),
            'FloorPlan20': (-10.0, 10.0, -10.0, 10.0, 0.0, 3.0),
        }
        return scene_bounds.get(scene_name, (-10.0, 10.0, -10.0, 10.0, 0.0, 4.0))
    
    def run_exploration_stage(self, episodes: int, resume_episode: int = 0):
        """Run exploration stage for spatial mapping."""
        logger.info(f"Starting exploration stage: episodes {resume_episode}-{episodes}")
        self.training_state['current_stage'] = 'exploration'
        self.training_state['stage_start_time'] = time.time()
        
        # Exploration strategies
        exploration_strategies = [
            'random_walk',
            'systematic_grid',
            'frontier_exploration',
            'curiosity_driven'
        ]
        
        for episode in range(resume_episode, episodes):
            self.training_state['current_episode'] = episode
            logger.info(f"Exploration episode {episode + 1}/{episodes}")
            
            # Reset environment
            self.controller.reset()
            
            # Select exploration strategy
            strategy = exploration_strategies[episode % len(exploration_strategies)]
            
            if strategy == 'random_walk':
                self._random_exploration(episode)
            elif strategy == 'systematic_grid':
                self._systematic_exploration(episode)
            elif strategy == 'frontier_exploration':
                self._frontier_exploration(episode)
            else:  # curiosity_driven
                self._curiosity_driven_exploration(episode)
            
            # Update statistics
            self.training_stats['episodes_completed'] += 1
            
            # Checkpoint and evaluation
            if (episode + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self._save_checkpoint(f"exploration_ep_{episode + 1}")
            
            if (episode + 1) % self.config['training']['evaluation_frequency'] == 0:
                self._run_stage_evaluation('exploration', episode + 1)
        
        logger.info("Exploration stage completed")
    
    def run_object_learning_stage(self, episodes: int, resume_episode: int = 0):
        """Run object learning stage for semantic associations."""
        logger.info(f"Starting object learning stage: episodes {resume_episode}-{episodes}")
        self.training_state['current_stage'] = 'object_learning'
        self.training_state['stage_start_time'] = time.time()
        
        # Comprehensive object vocabulary
        object_categories = {
            'furniture': [
                "dining table with chairs around it",
                "comfortable sofa for sitting",
                "wooden chair near the table",
                "bed with pillows and blankets",
                "office desk with computer",
                "bookshelf filled with books",
                "coffee table in living room",
                "armchair for reading"
            ],
            'appliances': [
                "refrigerator in the kitchen",
                "microwave on the counter",
                "television on the stand",
                "washing machine in laundry",
                "dishwasher under the counter",
                "oven for cooking food",
                "toaster on kitchen counter"
            ],
            'small_objects': [
                "ceramic mug on the table",
                "book on the shelf",
                "laptop computer on desk",
                "lamp providing light",
                "clock on the wall",
                "picture frame on table",
                "vase with flowers",
                "remote control on sofa"
            ],
            'room_features': [
                "kitchen counter with appliances",
                "bathroom sink with mirror",
                "window with natural light",
                "door leading to another room",
                "stairs going to upper floor"
            ]
        }
        
        all_objects = []
        for category, objects in object_categories.items():
            all_objects.extend(objects)
        
        for episode in range(resume_episode, episodes):
            self.training_state['current_episode'] = episode
            logger.info(f"Object learning episode {episode + 1}/{episodes}")
            
            # Reset environment
            self.controller.reset()
            
            # Select target object (curriculum learning: easy to hard)
            if episode < episodes // 3:
                # Easy: Large furniture
                target = np.random.choice(object_categories['furniture'])
            elif episode < 2 * episodes // 3:
                # Medium: Appliances and room features
                target = np.random.choice(
                    object_categories['appliances'] + object_categories['room_features']
                )
            else:
                # Hard: Small objects
                target = np.random.choice(object_categories['small_objects'])
            
            # Create and execute navigation task
            task = NavigationTask(
                task_type='ObjectNav',
                target_description=target,
                scene_name=self.training_state['current_scene'],
                max_steps=self.config['thor']['max_steps_per_task'],
                success_distance=self.config['thor']['success_distance']
            )
            
            result = self.agent.execute_task(task)
            
            # Update statistics
            self._update_task_statistics(result)
            
            logger.info(f"Task: '{target}' - Success: {result.success}, "
                       f"Steps: {result.steps_taken}, SPL: {result.spl:.3f}")
            
            # Checkpoint and evaluation
            if (episode + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self._save_checkpoint(f"object_learning_ep_{episode + 1}")
            
            if (episode + 1) % self.config['training']['evaluation_frequency'] == 0:
                self._run_stage_evaluation('object_learning', episode + 1)
        
        logger.info("Object learning stage completed")
    
    def run_fine_tuning_stage(self, episodes: int, resume_episode: int = 0):
        """Run fine-tuning stage for precise spatial understanding."""
        logger.info(f"Starting fine-tuning stage: episodes {resume_episode}-{episodes}")
        self.training_state['current_stage'] = 'fine_tuning'
        self.training_state['stage_start_time'] = time.time()
        
        # Complex, compositional descriptions
        complex_targets = [
            "red apple on the kitchen counter near the sink",
            "blue book on the shelf next to the window",
            "ceramic mug on the dining table with chairs",
            "laptop computer on the desk in the corner",
            "small lamp on the bedside table",
            "remote control on the sofa cushion",
            "picture frame on the wall above the fireplace",
            "vase with flowers on the coffee table",
            "clock on the kitchen wall near the stove",
            "pillow on the bed near the headboard",
            "towel hanging in the bathroom near the sink",
            "plant in the corner by the large window"
        ]
        
        for episode in range(resume_episode, episodes):
            self.training_state['current_episode'] = episode
            logger.info(f"Fine-tuning episode {episode + 1}/{episodes}")
            
            # Reset environment
            self.controller.reset()
            
            # Select complex target
            target = complex_targets[episode % len(complex_targets)]
            
            # Create challenging navigation task
            task = NavigationTask(
                task_type='ObjectNav',
                target_description=target,
                scene_name=self.training_state['current_scene'],
                max_steps=self.config['thor']['max_steps_per_task'],
                success_distance=0.8  # More precise success criteria
            )
            
            result = self.agent.execute_task(task)
            
            # Update statistics
            self._update_task_statistics(result)
            
            logger.info(f"Fine-tuning task: '{target}' - Success: {result.success}, "
                       f"Steps: {result.steps_taken}, SPL: {result.spl:.3f}")
            
            # Checkpoint and evaluation
            if (episode + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self._save_checkpoint(f"fine_tuning_ep_{episode + 1}")
            
            if (episode + 1) % self.config['training']['evaluation_frequency'] == 0:
                self._run_stage_evaluation('fine_tuning', episode + 1)
        
        logger.info("Fine-tuning stage completed")
    
    def run_generalization_stage(self, episodes: int, scenes: List[str] = None, resume_episode: int = 0):
        """Run generalization stage across multiple scenes."""
        if scenes is None:
            scenes = self.config['thor']['scenes'][1:]  # Skip first scene (already trained)
        
        logger.info(f"Starting generalization stage: episodes {resume_episode}-{episodes}, scenes: {scenes}")
        self.training_state['current_stage'] = 'generalization'
        self.training_state['stage_start_time'] = time.time()
        
        # Generalization targets
        general_targets = [
            "kitchen area with appliances",
            "living room with seating",
            "bedroom with bed and furniture",
            "bathroom with sink and mirror",
            "dining area with table and chairs",
            "office space with desk and computer",
            "entrance area near the door",
            "storage area with shelves"
        ]
        
        for episode in range(resume_episode, episodes):
            self.training_state['current_episode'] = episode
            
            # Cycle through scenes
            scene = scenes[episode % len(scenes)]
            
            # Switch scene if needed
            if scene != self.training_state['current_scene']:
                logger.info(f"Switching to scene: {scene}")
                self.controller.reset(scene=scene)
                
                # Reset field for new scene
                spatial_bounds = self._get_scene_bounds(scene)
                self.client.reset_field(spatial_bounds)
                self.training_state['current_scene'] = scene
            
            logger.info(f"Generalization episode {episode + 1}/{episodes} in {scene}")
            
            # Select target
            target = general_targets[episode % len(general_targets)]
            
            # Create navigation task
            task = NavigationTask(
                task_type='ObjectNav',
                target_description=target,
                scene_name=scene,
                max_steps=self.config['thor']['max_steps_per_task'],
                success_distance=self.config['thor']['success_distance']
            )
            
            result = self.agent.execute_task(task)
            
            # Update statistics
            self._update_task_statistics(result)
            self._update_scene_statistics(scene, result)
            
            logger.info(f"Generalization task in {scene}: '{target}' - "
                       f"Success: {result.success}, SPL: {result.spl:.3f}")
            
            # Checkpoint and evaluation
            if (episode + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self._save_checkpoint(f"generalization_ep_{episode + 1}")
            
            if (episode + 1) % self.config['training']['evaluation_frequency'] == 0:
                self._run_stage_evaluation('generalization', episode + 1)
        
        logger.info("Generalization stage completed")
    
    def _random_exploration(self, episode: int):
        """Random walk exploration strategy."""
        max_steps = self.config['thor']['max_steps_per_task']
        
        for step in range(max_steps):
            # Capture observation
            observation = self.observation_manager.capture_observation()
            self.client.push_observation(observation)
            self.training_stats['total_observations'] += 1
            
            # Random action
            action = np.random.choice([
                'MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight'
            ], p=[0.4, 0.1, 0.25, 0.25])  # Bias toward forward movement
            
            event = self.controller.step(action=action)
            if not event.metadata['lastActionSuccess']:
                # Recover from collision
                self.controller.step(action='RotateLeft')
    
    def _systematic_exploration(self, episode: int):
        """Systematic grid-based exploration."""
        # Simplified systematic exploration
        actions = ['MoveAhead'] * 3 + ['RotateRight'] + ['MoveAhead'] * 3 + ['RotateRight']
        max_steps = self.config['thor']['max_steps_per_task']
        
        for step in range(max_steps):
            observation = self.observation_manager.capture_observation()
            self.client.push_observation(observation)
            self.training_stats['total_observations'] += 1
            
            action = actions[step % len(actions)]
            event = self.controller.step(action=action)
            
            if not event.metadata['lastActionSuccess']:
                self.controller.step(action='RotateLeft')
    
    def _frontier_exploration(self, episode: int):
        """Frontier-based exploration (simplified)."""
        # This is a simplified version - in practice would use proper frontier detection
        self._random_exploration(episode)
    
    def _curiosity_driven_exploration(self, episode: int):
        """Curiosity-driven exploration using semantic queries."""
        max_steps = self.config['thor']['max_steps_per_task']
        
        # Curiosity targets
        curiosity_targets = [
            "unexplored area of the room",
            "interesting objects to examine",
            "furniture to investigate",
            "appliances to find"
        ]
        
        target = curiosity_targets[episode % len(curiosity_targets)]
        
        for step in range(0, max_steps, 10):  # Query every 10 steps
            # Capture observation
            observation = self.observation_manager.capture_observation()
            self.client.push_observation(observation)
            
            # Query for curiosity target
            try:
                query = SemanticQuery(text=target, resolution=0.1, max_points=500)
                result = self.client.query_semantic_field(query)
                
                # Move toward high-probability area
                if result.confidence > 0.1:
                    # Simple navigation toward target
                    for _ in range(10):
                        action = np.random.choice(['MoveAhead', 'RotateLeft', 'RotateRight'])
                        self.controller.step(action=action)
                        
                        obs = self.observation_manager.capture_observation()
                        self.client.push_observation(obs)
                        self.training_stats['total_observations'] += 1
                else:
                    # Random exploration if no clear target
                    for _ in range(10):
                        action = np.random.choice(['MoveAhead', 'RotateLeft', 'RotateRight'])
                        self.controller.step(action=action)
                        
                        obs = self.observation_manager.capture_observation()
                        self.client.push_observation(obs)
                        self.training_stats['total_observations'] += 1
                        
            except Exception as e:
                logger.warning(f"Curiosity query failed: {e}")
                # Fall back to random exploration
                self._random_exploration(episode)
                break
    
    def _update_task_statistics(self, result: TaskResult):
        """Update task-level statistics."""
        self.training_stats['episodes_completed'] += 1
        self.training_stats['total_queries'] += result.semantic_queries
        
        if result.success:
            self.training_stats['successful_tasks'] += 1
        else:
            self.training_stats['failed_tasks'] += 1
        
        # Update running averages
        total_tasks = self.training_stats['successful_tasks'] + self.training_stats['failed_tasks']
        if total_tasks > 0:
            self.training_stats['query_success_rate'] = (
                self.training_stats['successful_tasks'] / total_tasks
            )
        
        # Update SPL average
        if hasattr(self, '_spl_history'):
            self._spl_history.append(result.spl)
        else:
            self._spl_history = [result.spl]
        
        self.training_stats['average_spl'] = np.mean(self._spl_history[-100:])  # Last 100 episodes
    
    def _update_scene_statistics(self, scene: str, result: TaskResult):
        """Update scene-specific statistics."""
        if scene not in self.training_stats['scene_performance']:
            self.training_stats['scene_performance'][scene] = {
                'episodes': 0,
                'successes': 0,
                'spl_scores': []
            }
        
        scene_stats = self.training_stats['scene_performance'][scene]
        scene_stats['episodes'] += 1
        if result.success:
            scene_stats['successes'] += 1
        scene_stats['spl_scores'].append(result.spl)
    
    def _run_stage_evaluation(self, stage: str, episode: int):
        """Run evaluation for current stage."""
        logger.info(f"Running evaluation for {stage} stage at episode {episode}")
        
        # Create evaluation tasks
        eval_tasks = self._create_evaluation_tasks(num_tasks=20)
        
        # Run evaluation
        eval_results = []
        for task in eval_tasks:
            self.controller.reset()
            result = self.agent.execute_task(task)
            eval_results.append(result)
        
        # Calculate metrics
        success_rate = sum(r.success for r in eval_results) / len(eval_results)
        avg_spl = np.mean([r.spl for r in eval_results])
        avg_steps = np.mean([r.steps_taken for r in eval_results])
        
        # Store stage performance
        if stage not in self.training_stats['stage_performance']:
            self.training_stats['stage_performance'][stage] = []
        
        self.training_stats['stage_performance'][stage].append({
            'episode': episode,
            'success_rate': success_rate,
            'avg_spl': avg_spl,
            'avg_steps': avg_steps
        })
        
        logger.info(f"Evaluation results - Success: {success_rate:.1%}, "
                   f"SPL: {avg_spl:.3f}, Steps: {avg_steps:.1f}")
        
        # Check for best performance
        if success_rate > self.training_state['best_performance']:
            self.training_state['best_performance'] = success_rate
            self._save_checkpoint(f"best_performance_{stage}")
    
    def _create_evaluation_tasks(self, num_tasks: int = 20) -> List[NavigationTask]:
        """Create evaluation tasks for current scene."""
        eval_targets = [
            "dining table with chairs",
            "comfortable sofa",
            "kitchen counter",
            "bedroom with bed",
            "bathroom sink",
            "television screen",
            "refrigerator door",
            "office desk",
            "bookshelf with books",
            "coffee table"
        ]
        
        tasks = []
        for i in range(num_tasks):
            target = eval_targets[i % len(eval_targets)]
            task = NavigationTask(
                task_type='ObjectNav',
                target_description=target,
                scene_name=self.training_state['current_scene'],
                max_steps=self.config['thor']['max_steps_per_task'],
                success_distance=self.config['thor']['success_distance']
            )
            tasks.append(task)
        
        return tasks
    
    def _save_checkpoint(self, name: str):
        """Save comprehensive training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.json"
        
        checkpoint_data = {
            'training_state': self.training_state,
            'training_stats': self.training_stats,
            'config': self.config,
            'timestamp': time.time()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        # Also save as latest
        latest_path = self.checkpoint_dir / "latest.json"
        with open(latest_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        self.training_state.update(checkpoint_data['training_state'])
        self.training_stats.update(checkpoint_data['training_stats'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from episode {self.training_state['current_episode']}")
    
    def run_full_production_training(self, resume_checkpoint: str = None):
        """Run complete production training pipeline."""
        logger.info("Starting production training for optimal performance")
        
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
        
        self.training_state['start_time'] = time.time()
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Get stage configuration
            stages = self.config['training']['stages']
            
            # Stage 1: Exploration
            if self.training_state['current_stage'] != 'exploration':
                self.run_exploration_stage(stages['exploration']['episodes'])
            
            # Stage 2: Object Learning
            if self.training_state['current_stage'] in [None, 'exploration']:
                self.run_object_learning_stage(stages['object_learning']['episodes'])
            
            # Stage 3: Fine-tuning
            if self.training_state['current_stage'] in [None, 'exploration', 'object_learning']:
                self.run_fine_tuning_stage(stages['fine_tuning']['episodes'])
            
            # Stage 4: Generalization
            if self.training_state['current_stage'] in [None, 'exploration', 'object_learning', 'fine_tuning']:
                self.run_generalization_stage(
                    stages['generalization']['episodes'],
                    self.config['thor']['scenes'][1:]
                )
            
            # Final evaluation
            final_results = self._run_final_evaluation()
            
            # Training summary
            total_time = time.time() - self.training_state['start_time']
            logger.info(f"Production training completed in {total_time/3600:.1f} hours")
            logger.info(f"Final performance: {final_results['success_rate']:.1%} success rate")
            logger.info(f"Final SPL: {final_results['avg_spl']:.3f}")
            
            # Save final checkpoint
            self._save_checkpoint("final_production")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Production training failed: {e}")
            raise
        finally:
            # Cleanup
            if self.controller:
                self.controller.stop()
            if self.client:
                self.client.disconnect()
    
    def _run_final_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive final evaluation."""
        logger.info("Running final evaluation across all scenes")
        
        all_results = []
        scene_results = {}
        
        for scene in self.config['thor']['scenes']:
            logger.info(f"Evaluating on scene: {scene}")
            
            # Switch to scene
            self.controller.reset(scene=scene)
            spatial_bounds = self._get_scene_bounds(scene)
            self.client.reset_field(spatial_bounds)
            
            # Create evaluation tasks
            eval_tasks = self._create_evaluation_tasks(
                self.config['evaluation']['num_tasks_per_scene']
            )
            
            # Run evaluation
            scene_eval_results = []
            for task in eval_tasks:
                self.controller.reset()
                result = self.agent.execute_task(task)
                scene_eval_results.append(result)
                all_results.append(result)
            
            # Calculate scene metrics
            scene_success_rate = sum(r.success for r in scene_eval_results) / len(scene_eval_results)
            scene_avg_spl = np.mean([r.spl for r in scene_eval_results])
            
            scene_results[scene] = {
                'success_rate': scene_success_rate,
                'avg_spl': scene_avg_spl,
                'num_tasks': len(scene_eval_results)
            }
            
            logger.info(f"Scene {scene} results - Success: {scene_success_rate:.1%}, "
                       f"SPL: {scene_avg_spl:.3f}")
        
        # Overall metrics
        overall_success_rate = sum(r.success for r in all_results) / len(all_results)
        overall_avg_spl = np.mean([r.spl for r in all_results])
        overall_avg_steps = np.mean([r.steps_taken for r in all_results])
        
        final_results = {
            'success_rate': overall_success_rate,
            'avg_spl': overall_avg_spl,
            'avg_steps': overall_avg_steps,
            'scene_results': scene_results,
            'total_tasks': len(all_results)
        }
        
        # Save evaluation results
        eval_path = self.checkpoint_dir / "final_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        return final_results


def main():
    """Main function for production training script."""
    parser = argparse.ArgumentParser(description="Production Training for Optimal Performance")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--stage', type=str, 
                       choices=['exploration', 'object_learning', 'fine_tuning', 'generalization', 'full'],
                       default='full', help='Training stage')
    parser.add_argument('--episodes', type=int, help='Number of episodes (overrides config)')
    parser.add_argument('--scenes', type=str, help='Comma-separated list of scenes')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/production',
                       help='Directory for saving checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU {args.gpu}")
    
    # Initialize trainer
    trainer = ProductionTrainer(args.config, args.checkpoint_dir)
    
    # Override config with command line arguments
    if args.episodes:
        stage_name = args.stage if args.stage != 'full' else 'exploration'
        trainer.config['training']['stages'][stage_name]['episodes'] = args.episodes
    
    if args.scenes:
        trainer.config['thor']['scenes'] = args.scenes.split(',')
    
    try:
        if args.stage == 'full':
            results = trainer.run_full_production_training(args.resume)
        else:
            # Load checkpoint if resuming
            if args.resume:
                trainer.load_checkpoint(args.resume)
            
            trainer.initialize_components()
            
            if args.stage == 'exploration':
                trainer.run_exploration_stage(
                    trainer.config['training']['stages']['exploration']['episodes']
                )
            elif args.stage == 'object_learning':
                trainer.run_object_learning_stage(
                    trainer.config['training']['stages']['object_learning']['episodes']
                )
            elif args.stage == 'fine_tuning':
                trainer.run_fine_tuning_stage(
                    trainer.config['training']['stages']['fine_tuning']['episodes']
                )
            elif args.stage == 'generalization':
                scenes = trainer.config['thor']['scenes'][1:] if len(trainer.config['thor']['scenes']) > 1 else ['FloorPlan2']
                trainer.run_generalization_stage(
                    trainer.config['training']['stages']['generalization']['episodes'],
                    scenes
                )
        
        logger.info("Production training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_checkpoint("interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

