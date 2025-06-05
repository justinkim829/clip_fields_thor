"""
AI2-THOR Integration with CLIP-Fields Semantic Memory
====================================================

This module implements the AI2-THOR side of the integration, including
the navigation process, observation management, and task execution
components that leverage semantic memory for long-horizon navigation.
"""

import ai2thor
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import threading
from pathlib import Path

# Import bridge components
import sys
sys.path.append('../bridge')
from communication_bridge import (
    CLIPFieldsClient, Observation, SemanticQuery, QueryResult,
    AsyncObservationBuffer, CoordinateTransformer, PerformanceMonitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NavigationTask:
    """Definition of a navigation task."""
    task_type: str  # 'ObjectNav', 'Fetch', 'PickUp', etc.
    target_description: str  # Natural language description
    target_category: Optional[str] = None  # Traditional category (for comparison)
    scene_name: str = ""
    max_steps: int = 500
    success_distance: float = 1.0  # Distance threshold for success


@dataclass
class TaskResult:
    """Result of executing a navigation task."""
    success: bool
    steps_taken: int
    path_length: float
    spl: float  # Success-weighted Path Length
    goal_distance: float
    execution_time: float
    semantic_queries: int
    metadata: Dict[str, Any]


class ObservationManager:
    """Manages observation capture and processing from AI2-THOR."""

    def __init__(self, controller: Controller):
        self.controller = controller
        self.last_pose = None
        self.step_count = 0

    def capture_observation(self) -> Observation:
        """Capture current observation from AI2-THOR."""
        event = self.controller.last_event

        # Extract RGB image
        rgb = event.frame  # Shape: (H, W, 3)

        # Extract depth image
        depth = event.depth_frame  # Shape: (H, W)
        if depth is not None:
            # Convert to meters (AI2-THOR depth is in meters)
            depth = depth.astype(np.float32)
        else:
            # Create dummy depth if not available
            depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

        # Extract camera pose
        agent_metadata = event.metadata['agent']
        position = agent_metadata['position']
        rotation = agent_metadata['rotation']

        # Convert to 4x4 transformation matrix (Unity coordinates)
        pose = self._pose_from_metadata(position, rotation)

        # Extract camera intrinsics
        camera_intrinsics = self._extract_camera_intrinsics(event)

        # Create observation
        observation = Observation(
            rgb=rgb,
            depth=depth,
            pose=pose,
            timestamp=time.time(),
            camera_intrinsics=camera_intrinsics,
            metadata={
                'step_count': self.step_count,
                'scene_name': event.metadata.get('sceneName', ''),
                'agent_id': agent_metadata.get('agentId', 0)
            }
        )

        self.last_pose = pose
        self.step_count += 1

        return observation

    def _pose_from_metadata(self, position: Dict, rotation: Dict) -> np.ndarray:
        """Convert AI2-THOR position/rotation to 4x4 transformation matrix."""
        # Extract position
        x, y, z = position['x'], position['y'], position['z']

        # Extract rotation (Euler angles in degrees)
        rx, ry, rz = rotation['x'], rotation['y'], rotation['z']

        # Convert to radians
        rx, ry, rz = np.radians([rx, ry, rz])

        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Create 4x4 transformation matrix
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = [x, y, z]

        return pose

    def _extract_camera_intrinsics(self, event) -> Dict[str, float]:
        """Extract camera intrinsic parameters."""
        # AI2-THOR camera parameters
        # These are typical values - may need adjustment based on actual setup
        height, width = event.frame.shape[:2]

        # Assume 90-degree FOV (typical for AI2-THOR)
        fov_degrees = 90.0
        fov_radians = np.radians(fov_degrees)

        # Calculate focal length
        focal_length = (width / 2.0) / np.tan(fov_radians / 2.0)

        return {
            'fx': focal_length,
            'fy': focal_length,
            'cx': width / 2.0,
            'cy': height / 2.0,
            'width': width,
            'height': height
        }


class SemanticNavigationAgent:
    """Navigation agent that uses semantic memory for decision making."""

    def __init__(self, controller: Controller, semantic_client: CLIPFieldsClient):
        self.controller = controller
        self.semantic_client = semantic_client
        self.observation_manager = ObservationManager(controller)
        self.performance_monitor = PerformanceMonitor()

        # Navigation parameters
        self.actions = [
            'MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight',
            'LookUp', 'LookDown', 'Done'
        ]

        # Semantic query parameters
        self.query_frequency = 10  # Query every N steps
        self.last_query_step = 0

    def execute_task(self, task: NavigationTask) -> TaskResult:
        """Execute a navigation task using semantic memory."""
        logger.info(f"Starting task: {task.task_type} - {task.target_description}")

        start_time = time.time()
        steps_taken = 0
        path_length = 0.0
        semantic_queries = 0
        last_position = None

        # Initialize semantic field for this scene
        spatial_bounds = self._get_scene_bounds()
        self.semantic_client.reset_field(spatial_bounds)

        # Main navigation loop
        success = False
        while steps_taken < task.max_steps and not success:
            # Capture current observation
            observation = self.observation_manager.capture_observation()

            # Update semantic memory (asynchronous)
            self.semantic_client.push_observation(observation)

            # Check if we should query semantic memory
            if steps_taken - self.last_query_step >= self.query_frequency:
                target_location = self._query_target_location(task.target_description)
                semantic_queries += 1
                self.last_query_step = steps_taken
            else:
                target_location = None

            # Choose next action
            action = self._choose_action(observation, target_location, task)

            # Execute action
            if action == 'Done':
                success = self._check_task_success(task)
                break
            else:
                event = self.controller.step(action=action)

                # Update path length
                if last_position is not None:
                    current_position = event.metadata['agent']['position']
                    distance = np.sqrt(
                        (current_position['x'] - last_position['x'])**2 +
                        (current_position['z'] - last_position['z'])**2
                    )
                    path_length += distance

                last_position = event.metadata['agent']['position']
                steps_taken += 1

        # Calculate results
        execution_time = time.time() - start_time
        goal_distance = self._get_goal_distance(task)

        # Calculate SPL (Success-weighted Path Length)
        if success:
            optimal_path_length = self._estimate_optimal_path_length(task)
            spl = optimal_path_length / max(path_length, optimal_path_length)
        else:
            spl = 0.0

        result = TaskResult(
            success=success,
            steps_taken=steps_taken,
            path_length=path_length,
            spl=spl,
            goal_distance=goal_distance,
            execution_time=execution_time,
            semantic_queries=semantic_queries,
            metadata={
                'task_type': task.task_type,
                'target_description': task.target_description,
                'scene_name': task.scene_name
            }
        )

        logger.info(f"Task completed: Success={success}, Steps={steps_taken}, SPL={spl:.3f}")
        return result

    def _get_scene_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Estimate spatial bounds of the current scene."""
        # This is a simplified implementation
        # In practice, you might want to explore the scene or use scene metadata
        return (-10.0, 10.0, -10.0, 10.0, 0.0, 4.0)  # x_min, x_max, z_min, z_max, y_min, y_max

    def _query_target_location(self, target_description: str) -> Optional[Tuple[float, float, float]]:
        """Query semantic memory for target location."""
        try:
            query = SemanticQuery(
                text=target_description,
                resolution=0.1,  # 10cm resolution
                max_points=1000
            )

            result = self.semantic_client.query_semantic_field(query)

            if result.confidence > 0.1:  # Minimum confidence threshold
                # Convert from NeRF to Unity coordinates
                nerf_location = np.array(result.max_prob_location)
                unity_location = CoordinateTransformer.nerf_to_unity_point(nerf_location)
                return tuple(unity_location)
            else:
                logger.info(f"Low confidence ({result.confidence:.3f}) for query: {target_description}")
                return None

        except Exception as e:
            logger.error(f"Failed to query target location: {e}")
            return None

    def _choose_action(self, observation: Observation, target_location: Optional[Tuple[float, float, float]], task: NavigationTask) -> str:
        """Choose next action based on observation and semantic information."""
        # This is a simplified navigation policy
        # In practice, you would use more sophisticated planning

        if target_location is not None:
            # Navigate towards target location
            agent_pos = observation.pose[:3, 3]
            target_pos = np.array(target_location)

            # Calculate direction to target
            direction = target_pos - agent_pos
            distance = np.linalg.norm(direction)

            # Check if we're close enough to the target
            if distance < task.success_distance:
                return 'Done'

            # Simple navigation: turn towards target, then move forward
            direction_2d = direction[[0, 2]]  # x, z components
            direction_2d = direction_2d / np.linalg.norm(direction_2d)

            # Get current facing direction
            current_rotation = observation.pose[:3, :3]
            forward_direction = current_rotation[:, 2]  # Forward is -Z in Unity
            forward_2d = forward_direction[[0, 2]]
            forward_2d = forward_2d / np.linalg.norm(forward_2d)

            # Calculate angle between current direction and target direction
            dot_product = np.dot(forward_2d, direction_2d)
            cross_product = np.cross(forward_2d, direction_2d)

            # Decide action based on angle
            if abs(cross_product) > 0.3:  # Need to turn
                if cross_product > 0:
                    return 'RotateLeft'
                else:
                    return 'RotateRight'
            else:
                return 'MoveAhead'
        else:
            # Exploration behavior when no target location is known
            return np.random.choice(['MoveAhead', 'RotateLeft', 'RotateRight'])

    def _check_task_success(self, task: NavigationTask) -> bool:
        """Check if the current task has been completed successfully."""
        # This is a simplified success check
        # In practice, you would check for specific task completion criteria

        if task.task_type == 'ObjectNav':
            # Check if target object is visible and close
            goal_distance = self._get_goal_distance(task)
            return goal_distance < task.success_distance

        return False

    def _get_goal_distance(self, task: NavigationTask) -> float:
        """Get distance to goal (simplified implementation)."""
        # This would typically involve finding the actual target object
        # For now, return a placeholder value
        return 5.0

    def _estimate_optimal_path_length(self, task: NavigationTask) -> float:
        """Estimate optimal path length for SPL calculation."""
        # This would typically use A* or similar pathfinding
        # For now, return a placeholder value
        return 10.0


class TaskExecutor:
    """Executes and evaluates navigation tasks."""

    def __init__(self, controller: Controller, semantic_client: CLIPFieldsClient):
        self.controller = controller
        self.semantic_client = semantic_client
        self.agent = SemanticNavigationAgent(controller, semantic_client)

    def run_evaluation(self, tasks: List[NavigationTask]) -> List[TaskResult]:
        """Run evaluation on a list of tasks."""
        results = []

        for i, task in enumerate(tasks):
            logger.info(f"Running task {i+1}/{len(tasks)}: {task.target_description}")

            # Load scene if specified
            if task.scene_name:
                self._load_scene(task.scene_name)

            # Execute task
            result = self.agent.execute_task(task)
            results.append(result)

            # Reset agent position for next task
            self._reset_agent()

        return results

    def _load_scene(self, scene_name: str):
        """Load a specific scene."""
        try:
            self.controller.reset(scene=scene_name)
            logger.info(f"Loaded scene: {scene_name}")
        except Exception as e:
            logger.error(f"Failed to load scene {scene_name}: {e}")

    def _reset_agent(self):
        """Reset agent to starting position."""
        try:
            self.controller.reset()
        except Exception as e:
            logger.error(f"Failed to reset agent: {e}")


def create_sample_tasks() -> List[NavigationTask]:
    """Create sample navigation tasks for testing."""
    tasks = [
        NavigationTask(
            task_type='ObjectNav',
            target_description='red apple on the counter',
            scene_name='FloorPlan1'
        ),
        NavigationTask(
            task_type='ObjectNav',
            target_description='ceramic mug on the table',
            scene_name='FloorPlan1'
        ),
        NavigationTask(
            task_type='ObjectNav',
            target_description='blue book on the shelf',
            scene_name='FloorPlan1'
        )
    ]
    return tasks


def main():
    """Main function for testing the integration."""
    logger.info("Starting AI2-THOR CLIP-Fields integration test")

    # Initialize AI2-THOR controller
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan1",
        platform=CloudRendering,
        commit_id=None,
        local_build=False,
        branch="main"
    )

    # Initialize semantic client
    semantic_client = CLIPFieldsClient()

    # Create task executor
    executor = TaskExecutor(controller, semantic_client)

    # Create sample tasks
    tasks = create_sample_tasks()

    # Run evaluation
    results = executor.run_evaluation(tasks)

    # Print results
    print("\\nEvaluation Results:")
    print("=" * 50)
    for i, result in enumerate(results):
        print(f"Task {i+1}: {result.metadata['target_description']}")
        print(f"  Success: {result.success}")
        print(f"  Steps: {result.steps_taken}")
        print(f"  SPL: {result.spl:.3f}")
        print(f"  Semantic Queries: {result.semantic_queries}")
        print()

    # Calculate summary statistics
    success_rate = sum(r.success for r in results) / len(results)
    avg_spl = np.mean([r.spl for r in results])
    avg_steps = np.mean([r.steps_taken for r in results])

    print("Summary Statistics:")
    print(f"  Success Rate: {success_rate:.3f}")
    print(f"  Average SPL: {avg_spl:.3f}")
    print(f"  Average Steps: {avg_steps:.1f}")

    # Cleanup
    controller.stop()
    semantic_client.disconnect()


if __name__ == "__main__":
    main()

