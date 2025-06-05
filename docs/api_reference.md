# CLIP-Fields Thor Integration: API Reference

## Table of Contents

1. [Communication Bridge (`bridge/communication_bridge.py`)](#communication-bridge)
   - [`CLIPFieldsClient`](#clipfieldsclient)
   - [`CLIPFieldsServer`](#clipfieldsserver)
   - [`Observation`](#observation)
   - [`SemanticQuery`](#semanticquery)
   - [`QueryResult`](#queryresult)
   - [`DataCompressor`](#dataccompressor)
   - [`CoordinateTransformer`](#coordinatetransformer)
   - [`PerformanceMonitor`](#performancemonitor)
   - [`AsyncObservationBuffer`](#asyncobservationbuffer)
2. [AI2-THOR Integration (`thor_env/thor_integration.py`)](#ai2-thor-integration)
   - [`NavigationTask`](#navigationtask)
   - [`TaskResult`](#taskresult)
   - [`ObservationManager`](#observationmanager)
   - [`SemanticNavigationAgent`](#semanticnavigationagent)
   - [`TaskExecutor`](#taskexecutor)
3. [CLIP-Fields Model (`clipfields_env/clip_fields_model.py`)](#clip-fields-model)
   - [`CLIPFieldsModel`](#clipfieldsmodel)
   - [`WeakSupervisionPipeline`](#weaksupervisionpipeline)
   - [`CLIPFieldsTrainer`](#clipfieldstrainer)
   - [`CLIPFieldsInterface`](#clipfieldsinterface)

## Communication Bridge (`bridge/communication_bridge.py`)

This module defines the core components for communication between the AI2-THOR process and the CLIP-Fields process.

### `CLIPFieldsClient`

**Description**: Client class for interacting with the CLIP-Fields server via ZeroRPC.

**Initialization**:
```python
client = CLIPFieldsClient(server_address: str = "tcp://127.0.0.1:4242", timeout: int = 30, max_retries: int = 3)
```
- `server_address`: Network address of the CLIP-Fields server.
- `timeout`: Communication timeout in seconds.
- `max_retries`: Maximum number of connection retries.

**Methods**:
- `connect()`: Establishes connection to the server.
- `disconnect()`: Closes the connection.
- `push_observation(observation: Observation) -> bool`: Sends an observation to the server for field updates. Compresses data before sending.
- `query_semantic_field(query: SemanticQuery) -> QueryResult`: Sends a semantic query to the server and returns the result. Handles decompression of the result.
- `get_field_status() -> Dict[str, Any]`: Retrieves the current status of the semantic field from the server.
- `reset_field(spatial_bounds: Tuple[float, ...]) -> bool`: Requests the server to reset the semantic field with new spatial bounds.
- `get_performance_stats() -> Dict[str, float]`: Retrieves performance statistics from the client-side monitor.

### `CLIPFieldsServer`

**Description**: ZeroRPC server class that exposes CLIP-Fields functionality. Runs in the `clipfields` environment.

**Initialization**:
```python
# Internal initialization within server.py
server = zerorpc.Server(CLIPFieldsServer())
```

**Exposed Methods (via ZeroRPC)**:
- `push_observation(compressed_obs: Dict[str, Any]) -> bool`: Receives a compressed observation, decompresses it, and updates the field using `CLIPFieldsInterface`.
- `query_semantic_field(query_dict: Dict[str, Any]) -> Dict[str, Any]`: Receives a query dictionary, reconstructs `SemanticQuery`, queries the field, compresses the result, and returns it.
- `get_field_status() -> Dict[str, Any]`: Returns the status dictionary from `CLIPFieldsInterface`.
- `reset_field(spatial_bounds: Tuple[float, ...]) -> bool`: Resets the field using `CLIPFieldsInterface`.

### `Observation`

**Description**: Dataclass representing a single observation from AI2-THOR.

**Attributes**:
- `rgb: np.ndarray`: RGB image (H, W, 3), uint8.
- `depth: np.ndarray`: Depth map (H, W), float32, in meters.
- `pose: np.ndarray`: 4x4 camera pose matrix (world-to-camera, Unity coordinates).
- `timestamp: float`: Observation timestamp (Unix time).
- `camera_intrinsics: Dict[str, float]`: Camera intrinsic parameters (`fx`, `fy`, `cx`, `cy`, `width`, `height`).
- `metadata: Dict[str, Any]`: Additional metadata (e.g., step count, scene name).

### `SemanticQuery`

**Description**: Dataclass representing a semantic query.

**Attributes**:
- `text: str`: Natural language description of the target.
- `spatial_bounds: Optional[Tuple[float, ...]] = None`: Optional spatial bounds (xmin, xmax, zmin, zmax, ymin, ymax) to restrict the query.
- `resolution: float = 0.1`: Desired spatial resolution for the query result (in meters).
- `max_points: int = 1000`: Maximum number of points in the resulting probability map.

### `QueryResult`

**Description**: Dataclass representing the result of a semantic query.

**Attributes**:
- `query: str`: The original query text.
- `probability_map: np.ndarray`: 3D grid of probabilities (X, Y, Z), float32.
- `spatial_coords: np.ndarray`: Corresponding 3D coordinates for the probability map (X, Y, Z, 3), float32, NeRF coordinates.
- `max_prob_location: Tuple[float, float, float]`: 3D coordinates (NeRF) of the highest probability location.
- `confidence: float`: Maximum probability value in the map.
- `processing_time: float`: Time taken by the server to process the query (in milliseconds).

### `DataCompressor`

**Description**: Static class providing methods for compressing and decompressing observation and query data.

**Static Methods**:
- `compress_rgb(rgb: np.ndarray, quality: int = 90) -> bytes`: Compresses RGB image using JPEG.
- `decompress_rgb(compressed_rgb: bytes) -> np.ndarray`: Decompresses RGB image.
- `compress_depth(depth: np.ndarray) -> bytes`: Compresses depth map using zlib.
- `decompress_depth(compressed_depth: bytes, shape: Tuple[int, int]) -> np.ndarray`: Decompresses depth map.
- `compress_observation(observation: Observation) -> Dict[str, Any]`: Compresses all components of an `Observation` object.
- `decompress_observation(compressed_obs: Dict[str, Any]) -> Observation`: Decompresses data and reconstructs an `Observation` object.

### `CoordinateTransformer`

**Description**: Static class for converting coordinates between Unity (AI2-THOR) and NeRF (CLIP-Fields) systems.

**Static Methods**:
- `unity_to_nerf_pose(unity_pose: np.ndarray) -> np.ndarray`: Converts a 4x4 Unity pose matrix to NeRF pose matrix.
- `nerf_to_unity_pose(nerf_pose: np.ndarray) -> np.ndarray`: Converts a 4x4 NeRF pose matrix to Unity pose matrix.
- `unity_to_nerf_point(unity_point: np.ndarray) -> np.ndarray`: Converts a 3D point from Unity to NeRF coordinates.
- `nerf_to_unity_point(nerf_point: np.ndarray) -> np.ndarray`: Converts a 3D point from NeRF to Unity coordinates.

### `PerformanceMonitor`

**Description**: Class for monitoring client-side performance metrics (e.g., query latency).

**Initialization**:
```python
monitor = PerformanceMonitor(window_size: int = 100)
```
- `window_size`: Number of recent measurements to consider for statistics.

**Methods**:
- `record_query_time(duration_ms: float)`: Records the duration of a semantic query.
- `record_update_time(duration_ms: float)`: Records the duration of pushing an observation.
- `get_stats() -> Dict[str, float]`: Returns a dictionary of performance statistics (average, min, max, stddev for query and update times).

### `AsyncObservationBuffer`

**Description**: Class for buffering observations and sending them asynchronously to the server.

**Initialization**:
```python
buffer = AsyncObservationBuffer(max_size: int = 100)
```
- `max_size`: Maximum number of observations to buffer.

**Methods**:
- `start(client: CLIPFieldsClient)`: Starts the background worker thread.
- `stop()`: Stops the worker thread.
- `add_observation(observation: Observation)`: Adds an observation to the buffer.

## AI2-THOR Integration (`thor_env/thor_integration.py`)

This module implements the AI2-THOR side of the integration, including navigation agents and task execution logic.

### `NavigationTask`

**Description**: Dataclass defining a navigation task.

**Attributes**:
- `task_type: str`: Type of task (e.g., 'ObjectNav', 'Fetch').
- `target_description: str`: Natural language description of the target.
- `target_category: Optional[str]`: Optional traditional category for comparison.
- `scene_name: str`: Name of the AI2-THOR scene.
- `max_steps: int`: Maximum allowed steps for the task.
- `success_distance: float`: Distance threshold (meters) for success.

### `TaskResult`

**Description**: Dataclass holding the results of a completed navigation task.

**Attributes**:
- `success: bool`: Whether the task was completed successfully.
- `steps_taken: int`: Number of steps executed.
- `path_length: float`: Total distance traveled by the agent.
- `spl: float`: Success-weighted Path Length.
- `goal_distance: float`: Final distance to the goal.
- `execution_time: float`: Total time taken for the task.
- `semantic_queries: int`: Number of semantic queries performed.
- `metadata: Dict[str, Any]`: Additional task metadata.

### `ObservationManager`

**Description**: Manages capturing and processing observations from the AI2-THOR controller.

**Initialization**:
```python
manager = ObservationManager(controller: Controller)
```
- `controller`: An initialized AI2-THOR `Controller` instance.

**Methods**:
- `capture_observation() -> Observation`: Captures the current RGB, depth, pose, and intrinsics from the controller and returns an `Observation` object.

### `SemanticNavigationAgent`

**Description**: Embodied agent that uses the semantic memory client for navigation decisions.

**Initialization**:
```python
agent = SemanticNavigationAgent(controller: Controller, semantic_client: CLIPFieldsClient)
```
- `controller`: AI2-THOR `Controller` instance.
- `semantic_client`: Initialized `CLIPFieldsClient` instance.

**Methods**:
- `execute_task(task: NavigationTask) -> TaskResult`: Executes a given navigation task using semantic queries and a navigation policy.
- `_query_target_location(target_description: str) -> Optional[Tuple[float, ...]]`: Queries the semantic field for the target location.
- `_choose_action(...) -> str`: Selects the next navigation action based on current state and semantic information.
- `_check_task_success(task: NavigationTask) -> bool`: Evaluates if the task completion criteria are met.

### `TaskExecutor`

**Description**: Orchestrates the execution and evaluation of multiple navigation tasks.

**Initialization**:
```python
executor = TaskExecutor(controller: Controller, semantic_client: CLIPFieldsClient)
```
- `controller`: AI2-THOR `Controller` instance.
- `semantic_client`: Initialized `CLIPFieldsClient` instance.

**Methods**:
- `run_evaluation(tasks: List[NavigationTask]) -> List[TaskResult]`: Runs a list of navigation tasks and returns their results.
- `_load_scene(scene_name: str)`: Loads a specific scene in the controller.
- `_reset_agent()`: Resets the agent's position in the current scene.

## CLIP-Fields Model (`clipfields_env/clip_fields_model.py`)

This module contains the implementation of the CLIP-Fields model, training pipeline, and the interface exposed by the server.

### `CLIPFieldsModel`

**Description**: PyTorch `nn.Module` implementing the core CLIP-Fields neural network architecture.

**Initialization**:
```python
model = CLIPFieldsModel(
    spatial_bounds: Tuple[float, ...],
    num_levels: int = 16, level_dim: int = 8, ...,
    device: str = "cuda"
)
```
- `spatial_bounds`: Defines the 3D volume the field represents.
- Other parameters: Control the architecture of the hash grid encoder and MLP.
- `device`: Specifies the computation device ('cuda' or 'cpu').

**Methods**:
- `encode_spatial_location(coords: torch.Tensor) -> torch.Tensor`: Encodes 3D coordinates into semantic embeddings.
- `encode_text(text: str) -> torch.Tensor`: Encodes text into a joint embedding space.
- `encode_image_patch(image_patch: torch.Tensor) -> torch.Tensor`: Encodes an image patch into the joint embedding space.
- `query_field(text: str, spatial_coords: torch.Tensor) -> torch.Tensor`: Computes similarity between text and spatial embeddings at given coordinates.

### `WeakSupervisionPipeline`

**Description**: Handles the generation of training data (weak supervision) from AI2-THOR observations.

**Initialization**:
```python
pipeline = WeakSupervisionPipeline(device: str = "cuda")
```
- `device`: Computation device.

**Methods**:
- `process_observation(observation: Observation) -> Dict[str, Any]`: Takes an `Observation` and returns a dictionary containing 3D points, colors, semantic labels, and pose, suitable for training the `CLIPFieldsModel`.

### `CLIPFieldsTrainer`

**Description**: Manages the online training process for the `CLIPFieldsModel`.

**Initialization**:
```python
trainer = CLIPFieldsTrainer(model: CLIPFieldsModel, device: str = "cuda")
```
- `model`: The `CLIPFieldsModel` instance to train.
- `device`: Computation device.

**Methods**:
- `update_field(supervision_data: Dict[str, Any]) -> bool`: Performs one training step using the provided supervision data.
- `_compute_contrastive_loss(...) -> torch.Tensor`: Calculates the InfoNCE contrastive loss.
- `_update_memory_bank(...)`: Updates the memory bank used for negative sampling.

### `CLIPFieldsInterface`

**Description**: Main interface class used by the `CLIPFieldsServer` to interact with the model, trainer, and supervision pipeline.

**Initialization**:
```python
interface = CLIPFieldsInterface(spatial_bounds: Tuple[float, ...], device: str = "cuda")
```
- `spatial_bounds`: Initial spatial bounds for the field.
- `device`: Computation device.

**Methods**:
- `update_field(observation: Observation) -> bool`: Processes an observation, generates supervision, and updates the model.
- `query_field(query: SemanticQuery) -> QueryResult`: Performs a semantic query using the current model state.
- `get_status() -> Dict[str, Any]`: Returns the current status of the interface and model.
- `reset(spatial_bounds: Tuple[float, ...]) -> bool`: Resets the model and trainer with new spatial bounds.

