# CLIP-Fields Thor Integration

A comprehensive integration of CLIP-Fields neural semantic memory with AI2-THOR simulator for benchmarking long-horizon navigation tasks.

## Overview

This project implements a dual-process architecture that bridges CLIP-Fields' open-vocabulary spatial-semantic understanding with AI2-THOR's interactive 3D environments. The integration enables embodied agents to perform navigation tasks using natural language descriptions of targets, going beyond fixed object categories to support open-vocabulary understanding.

## Key Features

- **Open-Vocabulary Navigation**: Navigate to objects described with natural language (e.g., "ceramic mug on the table")
- **Spatial-Semantic Memory**: Learn persistent 3D semantic maps from RGB-D observations
- **Real-Time Performance**: Sub-2ms semantic queries suitable for 8Hz navigation control
- **Dual-Process Architecture**: Isolated environments for AI2-THOR and CLIP-Fields with ZeroRPC communication
- **Comprehensive Benchmarking**: Evaluation framework for comparing against baseline methods

## Architecture

```
┌─────────────────┐    ZeroRPC    ┌──────────────────┐
│   AI2-THOR      │◄─────────────►│   CLIP-Fields    │
│   Environment   │               │   Memory Process │
│   (Python 3.9)  │               │   (Python 3.8)   │
└─────────────────┘               └──────────────────┘
│                                 │
├─ Navigation Agent              ├─ Neural Field Model
├─ Observation Manager           ├─ Weak Supervision
├─ Task Executor                 ├─ Spatial Queries
└─ Coordinate Transform          └─ Online Learning
```

## Installation

### Prerequisites

- NVIDIA GPU with CUDA 11.8+ support
- Anaconda or Miniconda
- Git

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd clip_fields_thor_integration

# Run the automated setup script
./scripts/setup.sh
```

### Manual Setup

1. **Create AI2-THOR Environment**:
```bash
conda create -n thor python=3.9 -y
conda activate thor
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor
pip install -r thor_env/requirements.txt
```

2. **Create CLIP-Fields Environment**:
```bash
conda create -n clipfields python=3.8 -y
conda activate clipfields
conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.8 -c pytorch-lts -c nvidia -y
pip install -r clipfields_env/requirements.txt

# Install gridencoder CUDA extension
cd clipfields_env/gridencoder
export CUDA_HOME=/path/to/cuda
python setup.py install
```

## Usage

### Basic Usage

1. **Start the CLIP-Fields Server**:
```bash
conda activate clipfields
cd clipfields_env
python server.py
```

2. **Run Navigation Tasks**:
```bash
# In another terminal
conda activate thor
cd thor_env
python thor_integration.py
```

### Running the Demo

```bash
# Run complete demonstration
python examples/demo.py --demo-type full

# Run only semantic queries
python examples/demo.py --demo-type query

# Run only navigation tasks
python examples/demo.py --demo-type navigation
```

### Custom Tasks

```python
from thor_integration import NavigationTask, TaskExecutor
from communication_bridge import CLIPFieldsClient

# Create custom task
task = NavigationTask(
    task_type='ObjectNav',
    target_description='red apple on the kitchen counter',
    scene_name='FloorPlan1',
    max_steps=500
)

# Execute task
controller = Controller(scene="FloorPlan1")
client = CLIPFieldsClient()
executor = TaskExecutor(controller, client)
result = executor.agent.execute_task(task)
```

## Configuration

The system is configured through `configs/config.yaml`. Key settings include:

- **Communication**: Server address, timeouts, retry settings
- **AI2-THOR**: Scene settings, camera parameters, navigation parameters
- **CLIP-Fields**: Model architecture, training settings, spatial bounds
- **Evaluation**: Task types, metrics, baseline comparisons

## API Reference

### Core Components

#### CLIPFieldsClient
```python
client = CLIPFieldsClient(server_address="tcp://127.0.0.1:4242")

# Push observation for field updates
success = client.push_observation(observation)

# Query semantic field
query = SemanticQuery(text="ceramic mug", resolution=0.05)
result = client.query_semantic_field(query)
```

#### NavigationTask
```python
task = NavigationTask(
    task_type='ObjectNav',
    target_description='blue book on shelf',
    max_steps=500,
    success_distance=1.0
)
```

#### SemanticQuery
```python
query = SemanticQuery(
    text="target description",
    spatial_bounds=(-5, 5, -5, 5, 0, 3),  # Optional
    resolution=0.05,  # 5cm resolution
    max_points=1000
)
```

## Benchmarking

### Supported Tasks

- **ObjectNav**: Navigate to objects specified by natural language
- **Fetch**: Pick up and deliver objects to specified locations
- **PickUp**: Locate and pick up target objects

### Evaluation Metrics

- **Success Rate**: Percentage of successfully completed tasks
- **SPL (Success-weighted Path Length)**: Efficiency metric accounting for path optimality
- **Goal Distance**: Final distance to target location
- **Data Efficiency**: Performance vs. amount of training data

### Running Evaluations

```python
from thor_integration import TaskExecutor, create_sample_tasks

# Create evaluation tasks
tasks = create_sample_tasks()

# Run evaluation
executor = TaskExecutor(controller, semantic_client)
results = executor.run_evaluation(tasks)

# Analyze results
success_rate = sum(r.success for r in results) / len(results)
avg_spl = sum(r.spl for r in results) / len(results)
```

## Performance Optimization

### Real-Time Requirements

- Semantic queries: < 2ms for 1000 spatial points
- Field updates: Asynchronous, non-blocking
- Communication latency: < 10ms typical

### Memory Management

- Automatic observation buffering
- Configurable memory bank size
- GPU memory optimization

### Scaling

- Adaptive spatial bounds based on scene size
- Hierarchical spatial representations
- Distributed processing support

## Troubleshooting

### Common Issues

1. **CUDA Compilation Errors**:
   - Ensure CUDA_HOME is set correctly
   - Check PyTorch and CUDA version compatibility
   - Install ninja build system

2. **Connection Errors**:
   - Verify server is running on correct port
   - Check firewall settings
   - Ensure both environments can access the same network interface

3. **Memory Issues**:
   - Reduce batch size in config
   - Lower spatial resolution
   - Enable mixed precision training

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=/path/to/project
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python examples/demo.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 .
black .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{clipfields_thor_2024,
  title={Benchmarking Long-Horizon Navigation on THOR: CLIP-Fields},
  author={Your Name and Others},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [CLIP-Fields](https://github.com/notmahi/clip-fields) for the neural semantic field implementation
- [AI2-THOR](https://ai2thor.allenai.org/) for the simulation environment
- [SPOC](https://github.com/allenai/spoc-robot-training) for the navigation framework
- OpenAI CLIP for vision-language understanding

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com].

