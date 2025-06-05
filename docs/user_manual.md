# CLIP-Fields Thor Integration: User Manual

## Table of Contents

1. [Getting Started](#getting-started)
2. [System Architecture Overview](#system-architecture-overview)
3. [Basic Usage](#basic-usage)
4. [Advanced Configuration](#advanced-configuration)
5. [Creating Custom Tasks](#creating-custom-tasks)
6. [Benchmarking and Evaluation](#benchmarking-and-evaluation)
7. [Performance Optimization](#performance-optimization)
8. [Debugging and Monitoring](#debugging-and-monitoring)

## Getting Started

This user manual assumes you have successfully completed the installation process as described in the Installation Guide. Before proceeding, ensure that both the `thor` and `clipfields` Conda environments are properly set up and all dependencies are installed.

### Quick Start Example

The fastest way to get started with the CLIP-Fields Thor Integration is to run the provided demonstration script. This example will walk you through the basic functionality and help you understand how the system works.

1. **Open two terminal windows**. You will need one for the CLIP-Fields server and another for the AI2-THOR client.

2. **Start the CLIP-Fields server** in the first terminal:
   ```bash
   conda activate clipfields
   cd /path/to/clip_fields_thor_integration/clipfields_env
   python server.py
   ```

   You should see output indicating that the server has started successfully:
   ```
   INFO - CLIP-Fields server started at tcp://0.0.0.0:4242
   INFO - CLIP-Fields Model Interface initialized
   ```

3. **Run the demonstration** in the second terminal:
   ```bash
   conda activate thor
   cd /path/to/clip_fields_thor_integration
   python examples/demo.py --demo-type full
   ```

   This will run a complete demonstration that includes semantic queries and navigation tasks. The demo will automatically connect to the CLIP-Fields server, initialize AI2-THOR, and execute several example tasks.

### Understanding the Output

During the demonstration, you will see various types of output that help you understand what the system is doing:

**Server Output**: The CLIP-Fields server will log information about incoming observations and queries:
```
INFO - Updating field with observation at timestamp 1234567890.123
INFO - Querying field with text: ceramic mug on the table
```

**Client Output**: The AI2-THOR client will log navigation progress and task results:
```
INFO - Starting task: ObjectNav - ceramic mug on the table
INFO - Task completed: Success=True, Steps=45, SPL=0.823
```

**Performance Metrics**: The system will display performance statistics including query times, success rates, and path efficiency metrics.

## System Architecture Overview

Understanding the system architecture is crucial for effective use of the CLIP-Fields Thor Integration. The system consists of several key components that work together to enable semantic navigation.

### Dual-Process Architecture

The integration uses a dual-process architecture to handle the incompatible dependency requirements of AI2-THOR and CLIP-Fields. This design provides several advantages including dependency isolation, version flexibility, and improved reliability.

The **AI2-THOR Process** runs in the `thor` environment and handles all simulation-related operations. This process manages scene loading, agent control, observation capture, and task execution. It communicates with the CLIP-Fields process through a ZeroRPC bridge to access semantic memory capabilities.

The **CLIP-Fields Process** runs in the `clipfields` environment and provides spatial-semantic memory functionality. This process hosts the neural field model, performs online learning from observations, and responds to semantic queries. It operates independently of the simulation, allowing for flexible deployment and scaling.

### Communication Bridge

The ZeroRPC-based communication bridge enables efficient data exchange between the two processes while maintaining low latency suitable for real-time navigation. The bridge handles several types of operations including observation updates, semantic queries, field status monitoring, and system configuration.

**Observation Updates** are sent from the AI2-THOR process to the CLIP-Fields process whenever new RGB-D data is captured. These updates include compressed images, depth information, camera poses, and metadata. The CLIP-Fields process uses this information to continuously update the spatial-semantic field.

**Semantic Queries** allow the AI2-THOR process to search for objects or concepts within the learned semantic field. Queries are specified using natural language descriptions and return probability distributions over 3D space, indicating where the queried objects are likely to be located.

### Data Flow Pipeline

The data flow through the system follows a well-defined pipeline that ensures consistent and reliable operation. When the AI2-THOR agent moves through the environment, it continuously captures RGB-D observations along with precise pose information. These observations are processed through a coordinate transformation pipeline that converts from Unity's coordinate system to the NeRF-style coordinates used by CLIP-Fields.

The weak supervision pipeline generates semantic labels for the observations using pre-trained vision-language models. Object detection is performed using a domain-adapted version of Detic, while CLIP provides visual features and Sentence-BERT generates text embeddings. This supervision information is used to train the neural field to associate spatial locations with semantic content.

## Basic Usage

### Starting the System

The CLIP-Fields Thor Integration requires both processes to be running for full functionality. Always start the CLIP-Fields server before attempting to run navigation tasks.

1. **Start the CLIP-Fields Server**:
   ```bash
   conda activate clipfields
   cd clipfields_env
   python server.py
   ```

   The server will initialize the neural field model and begin listening for connections. You can customize the server configuration by modifying the `configs/config.yaml` file or by passing command-line arguments.

2. **Verify Server Status**:
   You can check that the server is running correctly by testing the connection:
   ```bash
   conda activate thor
   python -c "
   from bridge.communication_bridge import CLIPFieldsClient
   client = CLIPFieldsClient()
   print(client.get_field_status())
   client.disconnect()
   "
   ```

### Running Navigation Tasks

Once the server is running, you can execute navigation tasks using the AI2-THOR integration. The system supports several types of tasks, each with different objectives and evaluation criteria.

**ObjectNav Tasks** require the agent to navigate to a specific object described using natural language. These tasks test the system's ability to understand object descriptions and locate them within the environment:

```python
from thor_env.thor_integration import NavigationTask, TaskExecutor
from bridge.communication_bridge import CLIPFieldsClient
from ai2thor.controller import Controller

# Initialize components
controller = Controller(scene="FloorPlan1")
client = CLIPFieldsClient()
executor = TaskExecutor(controller, client)

# Create navigation task
task = NavigationTask(
    task_type='ObjectNav',
    target_description='red apple on the kitchen counter',
    scene_name='FloorPlan1',
    max_steps=500,
    success_distance=1.0
)

# Execute task
result = executor.agent.execute_task(task)
print(f"Task success: {result.success}")
print(f"Steps taken: {result.steps_taken}")
print(f"SPL: {result.spl:.3f}")
```

**Fetch Tasks** extend ObjectNav by requiring the agent to pick up the target object and deliver it to a specified location. These tasks test both navigation and manipulation capabilities:

```python
fetch_task = NavigationTask(
    task_type='Fetch',
    target_description='ceramic mug from the table to the kitchen sink',
    scene_name='FloorPlan1',
    max_steps=800
)
```

### Semantic Querying

One of the key capabilities of the system is the ability to perform semantic queries that return spatial probability distributions. This functionality can be used independently of navigation tasks for analysis and debugging.

```python
from bridge.communication_bridge import CLIPFieldsClient, SemanticQuery

# Connect to server
client = CLIPFieldsClient()

# Create semantic query
query = SemanticQuery(
    text="ceramic mug on the table",
    resolution=0.05,  # 5cm resolution
    max_points=1000
)

# Execute query
result = client.query_semantic_field(query)

print(f"Query: {result.query}")
print(f"Max probability location: {result.max_prob_location}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Processing time: {result.processing_time:.1f}ms")

# The result contains a 3D probability map
print(f"Probability map shape: {result.probability_map.shape}")
print(f"Spatial coordinates shape: {result.spatial_coords.shape}")
```

### Scene Management

The system supports multiple AI2-THOR scenes, and you can switch between them as needed for different experiments or evaluations.

```python
# Load a specific scene
controller.reset(scene="FloorPlan5")

# Reset the semantic field for the new scene
spatial_bounds = (-8.0, 8.0, -8.0, 8.0, 0.0, 3.0)
client.reset_field(spatial_bounds)

# The field will now learn the new scene as the agent explores
```

## Advanced Configuration

The system behavior can be extensively customized through the configuration file located at `configs/config.yaml`. This section describes the key configuration options and their effects on system performance.

### Communication Settings

The communication bridge can be configured to optimize performance for different network conditions and hardware setups:

```yaml
system:
  server_address: "tcp://127.0.0.1:4242"
  timeout: 30
  max_retries: 3
  async_buffer_size: 100
```

**Server Address**: Specifies the network address where the CLIP-Fields server listens for connections. For local deployment, use `127.0.0.1` or `localhost`. For distributed deployment across multiple machines, use the appropriate IP address.

**Timeout and Retries**: Control how the client handles connection failures and slow responses. Increase these values for unreliable network conditions or when running on slower hardware.

**Async Buffer Size**: Determines how many observations can be queued for processing. Larger buffers provide better performance but use more memory.

### AI2-THOR Configuration

The AI2-THOR simulation environment can be configured to match your experimental requirements:

```yaml
thor:
  default_scene: "FloorPlan1"
  agent_mode: "default"
  visibility_distance: 1.5
  grid_size: 0.25
  camera_fov: 90.0
  image_width: 224
  image_height: 224
```

**Scene Settings**: Control which scene is loaded by default and how the agent interacts with the environment. The `visibility_distance` parameter affects how far the agent can see objects, while `grid_size` determines the precision of agent movement.

**Camera Settings**: Configure the agent's visual sensor. Higher resolution images provide more detail but require more processing time and memory. The field of view affects how much of the environment is visible in each observation.

### CLIP-Fields Model Configuration

The neural field model architecture can be tuned for different trade-offs between accuracy, memory usage, and computational performance:

```yaml
clip_fields:
  num_levels: 16
  level_dim: 8
  base_resolution: 16
  log2_hashmap_size: 24
  mlp_depth: 2
  mlp_width: 256
```

**Hash Grid Parameters**: The `num_levels`, `level_dim`, and `base_resolution` parameters control the multi-resolution hash table that stores spatial features. More levels and higher dimensions provide greater representational capacity but require more memory.

**MLP Architecture**: The `mlp_depth` and `mlp_width` parameters define the neural network that processes spatial features. Deeper and wider networks can learn more complex relationships but are slower to train and evaluate.

### Training and Learning Configuration

The online learning behavior can be adjusted to balance between adaptation speed and stability:

```yaml
clip_fields:
  learning_rate: 0.002
  online_learning_rate: 0.0002
  consolidation_penalty: 0.01
  update_frequency: 10
```

**Learning Rates**: The base learning rate is used during initial training, while the online learning rate is used for updates during navigation. Lower online learning rates provide more stable learning but slower adaptation.

**Consolidation Penalty**: Helps prevent catastrophic forgetting by encouraging the model to preserve previously learned information. Higher values provide more stability but may slow adaptation to new information.

**Update Frequency**: Controls how often the field is updated with new observations. More frequent updates provide faster learning but may impact navigation performance.

## Creating Custom Tasks

The system is designed to be extensible, allowing you to create custom navigation tasks that suit your specific research needs. This section provides detailed guidance on implementing new task types and evaluation metrics.

### Defining Custom Task Types

Custom tasks are created by extending the `NavigationTask` class and implementing the necessary evaluation logic. Here's an example of creating a custom exploration task:

```python
from thor_env.thor_integration import NavigationTask, TaskResult
from typing import Dict, Any
import numpy as np

class ExplorationTask(NavigationTask):
    """Custom task that requires exploring a percentage of the environment."""
    
    def __init__(self, target_coverage: float = 0.8, **kwargs):
        super().__init__(task_type='Exploration', **kwargs)
        self.target_coverage = target_coverage
        self.visited_locations = set()
        
    def update_progress(self, agent_position: Dict[str, float]):
        """Update exploration progress based on agent position."""
        # Discretize position to grid cells
        grid_x = int(agent_position['x'] / 0.5)  # 0.5m grid
        grid_z = int(agent_position['z'] / 0.5)
        self.visited_locations.add((grid_x, grid_z))
        
    def check_success(self, total_navigable_area: float) -> bool:
        """Check if exploration target has been reached."""
        explored_area = len(self.visited_locations) * 0.25  # 0.5m^2 per cell
        coverage = explored_area / total_navigable_area
        return coverage >= self.target_coverage
```

### Implementing Custom Evaluation Metrics

You can define custom metrics to evaluate task performance according to your research objectives:

```python
def calculate_exploration_efficiency(result: TaskResult, task: ExplorationTask) -> float:
    """Calculate exploration efficiency metric."""
    coverage_achieved = len(task.visited_locations) * 0.25
    optimal_coverage = task.target_coverage * task.total_area
    
    if result.success:
        efficiency = optimal_coverage / coverage_achieved
    else:
        efficiency = coverage_achieved / (task.max_steps * 0.25)  # Normalize by max possible
    
    return min(efficiency, 1.0)

def calculate_semantic_query_efficiency(result: TaskResult) -> float:
    """Calculate efficiency of semantic query usage."""
    if result.semantic_queries == 0:
        return 0.0
    
    # Reward fewer queries for successful tasks
    if result.success:
        return 1.0 / (1.0 + 0.1 * result.semantic_queries)
    else:
        return 0.0
```

### Custom Scene Generation

For specialized experiments, you may want to create custom scenes with specific object arrangements or layouts:

```python
from ai2thor.controller import Controller
import json

def create_custom_scene(controller: Controller, object_config: Dict[str, Any]):
    """Create a custom scene with specified object placements."""
    
    # Start with a base scene
    controller.reset(scene="FloorPlan1")
    
    # Remove existing objects
    for obj in controller.last_event.metadata['objects']:
        if obj['moveable']:
            controller.step(
                action="RemoveFromScene",
                objectId=obj['objectId']
            )
    
    # Add custom objects
    for obj_type, positions in object_config.items():
        for i, pos in enumerate(positions):
            controller.step(
                action="CreateObject",
                objectType=obj_type,
                position=pos,
                rotation={'x': 0, 'y': 0, 'z': 0}
            )
    
    return controller.last_event.metadata

# Example usage
custom_objects = {
    'Apple': [
        {'x': 1.0, 'y': 1.0, 'z': 2.0},
        {'x': -1.0, 'y': 1.0, 'z': 1.0}
    ],
    'Mug': [
        {'x': 0.5, 'y': 0.9, 'z': -1.0}
    ]
}

scene_metadata = create_custom_scene(controller, custom_objects)
```

## Benchmarking and Evaluation

The system includes a comprehensive benchmarking framework that enables systematic evaluation of navigation performance and comparison with baseline methods. This section describes how to conduct rigorous experiments and analyze results.

### Standard Evaluation Protocol

The benchmarking framework follows established protocols from the embodied AI literature to ensure fair and reproducible comparisons. The evaluation process consists of several phases including task generation, execution, and analysis.

**Task Generation**: Tasks are generated using stratified sampling to ensure representative coverage of different difficulty levels, object types, and spatial configurations. The system supports both predefined task sets and dynamic generation based on scene content.

```python
from thor_env.thor_integration import TaskExecutor, create_sample_tasks
from evaluation.benchmark import BenchmarkSuite

# Create benchmark suite
benchmark = BenchmarkSuite(
    scenes=['FloorPlan1', 'FloorPlan2', 'FloorPlan5'],
    task_types=['ObjectNav', 'Fetch'],
    num_tasks_per_scene=50
)

# Generate evaluation tasks
tasks = benchmark.generate_tasks()
print(f"Generated {len(tasks)} evaluation tasks")
```

**Task Execution**: Tasks are executed in a controlled environment with consistent initialization and termination criteria. The system automatically handles scene loading, agent initialization, and result collection.

```python
# Initialize evaluation components
controller = Controller()
client = CLIPFieldsClient()
executor = TaskExecutor(controller, client)

# Run evaluation
results = []
for task in tasks:
    result = executor.agent.execute_task(task)
    results.append(result)
    
    # Log progress
    if len(results) % 10 == 0:
        print(f"Completed {len(results)}/{len(tasks)} tasks")
```

**Result Analysis**: The framework provides comprehensive analysis tools for computing standard metrics and generating visualizations.

### Performance Metrics

The system computes several standard metrics that are widely used in the embodied AI literature:

**Success Rate (SR)**: The percentage of tasks completed successfully within the specified step limit. This is the most fundamental metric for evaluating navigation performance.

```python
def calculate_success_rate(results: List[TaskResult]) -> float:
    successful_tasks = sum(1 for r in results if r.success)
    return successful_tasks / len(results)
```

**Success-weighted Path Length (SPL)**: A metric that combines success rate with path efficiency, penalizing agents that take unnecessarily long paths to reach their goals.

```python
def calculate_spl(results: List[TaskResult]) -> float:
    spl_scores = []
    for result in results:
        if result.success:
            optimal_path = estimate_optimal_path_length(result)
            spl = optimal_path / max(result.path_length, optimal_path)
        else:
            spl = 0.0
        spl_scores.append(spl)
    
    return np.mean(spl_scores)
```

**Data Efficiency**: Measures how quickly the system learns to perform tasks as a function of the amount of training data.

```python
def calculate_data_efficiency(results_by_data_amount: Dict[int, List[TaskResult]]) -> Dict[int, float]:
    efficiency_curve = {}
    for data_amount, results in results_by_data_amount.items():
        success_rate = calculate_success_rate(results)
        efficiency_curve[data_amount] = success_rate
    
    return efficiency_curve
```

### Baseline Comparisons

The benchmarking framework includes implementations of several baseline methods to provide context for evaluating the CLIP-Fields approach:

**Map-Free Baseline**: A reactive navigation policy that does not maintain any spatial memory. This baseline helps quantify the benefit of spatial-semantic memory.

**Fixed-Label Baseline**: Uses traditional object detection with a fixed vocabulary of object categories. This baseline isolates the contribution of open-vocabulary understanding.

**Oracle Baseline**: Has perfect knowledge of object locations within the environment. This baseline provides an upper bound on achievable performance.

```python
from evaluation.baselines import MapFreeBaseline, FixedLabelBaseline, OracleBaseline

# Run baseline comparisons
baselines = {
    'map_free': MapFreeBaseline(),
    'fixed_labels': FixedLabelBaseline(),
    'oracle': OracleBaseline()
}

baseline_results = {}
for name, baseline in baselines.items():
    print(f"Running {name} baseline...")
    baseline_results[name] = baseline.evaluate(tasks)
```

### Statistical Analysis

Proper statistical analysis is crucial for drawing valid conclusions from experimental results. The framework includes tools for significance testing and confidence interval estimation.

```python
from scipy import stats
import numpy as np

def compare_methods(results_a: List[TaskResult], results_b: List[TaskResult]) -> Dict[str, Any]:
    """Compare two methods using statistical tests."""
    
    # Extract success indicators
    success_a = [1 if r.success else 0 for r in results_a]
    success_b = [1 if r.success else 0 for r in results_b]
    
    # Perform statistical tests
    t_stat, p_value = stats.ttest_ind(success_a, success_b)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((np.std(success_a)**2 + np.std(success_b)**2) / 2))
    cohens_d = (np.mean(success_a) - np.mean(success_b)) / pooled_std
    
    return {
        'mean_difference': np.mean(success_a) - np.mean(success_b),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }
```

## Performance Optimization

Optimizing the performance of the CLIP-Fields Thor Integration is essential for conducting large-scale experiments and achieving real-time navigation performance. This section provides detailed guidance on identifying bottlenecks and implementing optimizations.

### Profiling and Monitoring

The first step in optimization is understanding where the system spends its time and resources. The integration includes built-in profiling tools that help identify performance bottlenecks.

```python
from bridge.communication_bridge import PerformanceMonitor
import time

# Initialize performance monitor
monitor = PerformanceMonitor(window_size=100)

# Monitor semantic queries
start_time = time.time()
result = client.query_semantic_field(query)
query_time = (time.time() - start_time) * 1000
monitor.record_query_time(query_time)

# Get performance statistics
stats = monitor.get_stats()
print(f"Average query time: {stats['avg_query_time']:.1f}ms")
print(f"Maximum query time: {stats['max_query_time']:.1f}ms")
```

### GPU Memory Optimization

The neural field model can consume significant GPU memory, especially for large scenes or high-resolution spatial representations. Several strategies can help reduce memory usage:

**Batch Size Tuning**: Reduce the batch size for field updates to lower peak memory usage:

```yaml
clip_fields:
  batch_size: 16  # Reduced from default 32
```

**Mixed Precision Training**: Enable mixed precision to reduce memory usage and improve performance:

```yaml
hardware:
  mixed_precision: true
  gpu_memory_fraction: 0.8
```

**Spatial Resolution Adjustment**: Use lower spatial resolution for initial exploration, then increase resolution for fine-grained queries:

```python
# Coarse exploration phase
coarse_query = SemanticQuery(
    text="kitchen area",
    resolution=0.2,  # 20cm resolution
    max_points=500
)

# Fine-grained localization phase
fine_query = SemanticQuery(
    text="ceramic mug on counter",
    resolution=0.05,  # 5cm resolution
    spatial_bounds=coarse_result.get_high_probability_region()
)
```

### Communication Optimization

The ZeroRPC communication bridge can be optimized to reduce latency and improve throughput:

**Image Compression**: Adjust compression settings based on available bandwidth:

```python
# High quality for detailed analysis
compressed_rgb = DataCompressor.compress_rgb(rgb, quality=95)

# Lower quality for real-time navigation
compressed_rgb = DataCompressor.compress_rgb(rgb, quality=75)
```

**Asynchronous Processing**: Use asynchronous observation processing to avoid blocking navigation:

```python
from bridge.communication_bridge import AsyncObservationBuffer

# Initialize async buffer
buffer = AsyncObservationBuffer(max_size=50)
buffer.start(client)

# Add observations without blocking
for observation in observation_stream:
    buffer.add_observation(observation)
    # Continue with navigation immediately
```

### Algorithmic Optimizations

Several algorithmic optimizations can improve the efficiency of semantic queries and field updates:

**Hierarchical Spatial Queries**: Use a coarse-to-fine approach for spatial queries:

```python
def hierarchical_query(client: CLIPFieldsClient, text: str, bounds: Tuple) -> QueryResult:
    """Perform hierarchical spatial query for improved efficiency."""
    
    # Coarse query to identify regions of interest
    coarse_query = SemanticQuery(
        text=text,
        spatial_bounds=bounds,
        resolution=0.2,
        max_points=200
    )
    coarse_result = client.query_semantic_field(coarse_query)
    
    # Identify high-probability regions
    high_prob_regions = extract_high_probability_regions(coarse_result)
    
    # Fine query in promising regions
    fine_results = []
    for region_bounds in high_prob_regions:
        fine_query = SemanticQuery(
            text=text,
            spatial_bounds=region_bounds,
            resolution=0.05,
            max_points=500
        )
        fine_result = client.query_semantic_field(fine_query)
        fine_results.append(fine_result)
    
    # Combine results
    return combine_query_results(fine_results)
```

**Adaptive Update Frequency**: Adjust field update frequency based on exploration progress:

```python
class AdaptiveUpdateScheduler:
    def __init__(self, base_frequency: int = 10):
        self.base_frequency = base_frequency
        self.exploration_progress = 0.0
        
    def should_update(self, step: int) -> bool:
        # Update more frequently during initial exploration
        if self.exploration_progress < 0.3:
            frequency = self.base_frequency // 2
        elif self.exploration_progress < 0.7:
            frequency = self.base_frequency
        else:
            frequency = self.base_frequency * 2
            
        return step % frequency == 0
```

## Debugging and Monitoring

Effective debugging and monitoring are essential for maintaining system reliability and diagnosing issues during development and deployment. The integration provides comprehensive logging and debugging tools.

### Logging Configuration

The system uses Python's standard logging framework with configurable levels and outputs:

```python
import logging

# Configure logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clip_fields_thor.log'),
        logging.StreamHandler()
    ]
)

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clip_fields_thor.log')
    ]
)
```

### System Health Monitoring

Monitor key system metrics to ensure optimal performance:

```python
import psutil
import torch

def monitor_system_health():
    """Monitor system resource usage and performance."""
    
    # CPU and memory usage
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # GPU usage (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
    else:
        gpu_memory = gpu_memory_max = 0
    
    # Log system status
    logger.info(f"System Health - CPU: {cpu_percent:.1f}%, "
                f"RAM: {memory.percent:.1f}%, "
                f"GPU Memory: {gpu_memory:.1f}/{gpu_memory_max:.1f}GB")
    
    # Check for potential issues
    if cpu_percent > 90:
        logger.warning("High CPU usage detected")
    if memory.percent > 90:
        logger.warning("High memory usage detected")
    if gpu_memory > 0.9 * gpu_memory_max:
        logger.warning("High GPU memory usage detected")
```

### Debugging Tools

The system includes several debugging tools to help diagnose issues:

**Visualization Tools**: Visualize semantic fields and query results:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_semantic_field(result: QueryResult, save_path: str = None):
    """Visualize semantic query results."""
    
    # Create 2D projection of 3D probability map
    prob_2d = np.max(result.probability_map, axis=1)  # Max over Y axis
    
    plt.figure(figsize=(10, 8))
    plt.imshow(prob_2d, cmap='hot', interpolation='bilinear')
    plt.colorbar(label='Probability')
    plt.title(f'Semantic Field: "{result.query}"')
    plt.xlabel('X coordinate')
    plt.ylabel('Z coordinate')
    
    # Mark maximum probability location
    max_loc = result.max_prob_location
    plt.plot(max_loc[0], max_loc[2], 'b*', markersize=15, label='Max Probability')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

**Connection Testing**: Test communication between processes:

```python
def test_communication_bridge():
    """Test the communication bridge functionality."""
    
    try:
        # Test connection
        client = CLIPFieldsClient()
        status = client.get_field_status()
        print(f"✓ Connection successful: {status}")
        
        # Test observation push
        dummy_obs = create_dummy_observation()
        success = client.push_observation(dummy_obs)
        print(f"✓ Observation push: {success}")
        
        # Test semantic query
        query = SemanticQuery(text="test object")
        result = client.query_semantic_field(query)
        print(f"✓ Semantic query: {result.processing_time:.1f}ms")
        
        client.disconnect()
        print("✓ All communication tests passed")
        
    except Exception as e:
        print(f"✗ Communication test failed: {e}")
```

This comprehensive user manual provides the foundation for effectively using the CLIP-Fields Thor Integration. The system's modular design and extensive configuration options enable adaptation to a wide range of research scenarios and experimental requirements.

