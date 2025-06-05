# CLIP-Fields Thor Integration - Complete Command Reference


## **1. IMMEDIATE TESTING (0 training time - Mock Components)**

```bash
# Terminal 1: Start server with mock components
conda activate clipfields
cd clip_fields_thor_integration/clipfields_env
export CLIP_AVAILABLE=False  # Use mock components
python server.py

# Terminal 2: Run immediate tests
conda activate thor
cd clip_fields_thor_integration

# Test semantic queries only (no navigation)
python examples/demo.py --demo-type query

# Test navigation logic with mock semantic responses
python examples/demo.py --demo-type navigation

# Full demo with mock components
python examples/demo.py --demo-type full
```

---

## **2. DEMO-READY TRAINING (2-4 hours)**

### **Option A: Automated Demo Training**
```bash
# Single command for complete demo-ready training
conda activate thor
cd clip_fields_thor_integration
python scripts/train_demo_ready.py --scene FloorPlan1 --hours 3 --auto-test
```

### **Option B: Manual Step-by-Step Training**
```bash
# Step 1: Start CLIP-Fields server
conda activate clipfields
cd clip_fields_thor_integration/clipfields_env
python server.py

# Step 2: Run rapid training (Terminal 2)
conda activate thor
cd clip_fields_thor_integration/thor_env

# Quick exploration training (1-2 hours)
python rapid_training.py --mode exploration --episodes 50 --scene FloorPlan1

# Object association training (1-2 hours)
python rapid_training.py --mode object_learning --episodes 50 --scene FloorPlan1

# Validation
python rapid_training.py --mode validation --episodes 20 --scene FloorPlan1

# Or run complete rapid training pipeline
python rapid_training.py --mode full --scene FloorPlan1
```

### **Quick Evaluation**
```bash
# Test demo readiness
python scripts/quick_eval.py --tasks 10 --scene FloorPlan1
```

---

## **3. OPTIMAL PERFORMANCE TRAINING (40-80 hours)**

### **Single Scene Production Training**
```bash
# Step 1: Start production server
conda activate clipfields
cd clip_fields_thor_integration/clipfields_env
python server.py

# Step 2: Run production training (Terminal 2)
conda activate thor
cd clip_fields_thor_integration/thor_env

# Complete production training pipeline
python production_training.py \
  --stage full \
  --scenes FloorPlan1,FloorPlan2,FloorPlan5 \
  --checkpoint-dir ../checkpoints/production

# Or run individual stages:

# Stage 1: Exploration (10-15 hours)
python production_training.py \
  --stage exploration \
  --episodes 200 \
  --scenes FloorPlan1 \
  --checkpoint-dir ../checkpoints/exploration

# Stage 2: Object learning (15-25 hours)
python production_training.py \
  --stage object_learning \
  --episodes 400 \
  --resume ../checkpoints/exploration/latest.json \
  --checkpoint-dir ../checkpoints/object_learning

# Stage 3: Fine-tuning (15-25 hours)
python production_training.py \
  --stage fine_tuning \
  --episodes 400 \
  --resume ../checkpoints/object_learning/latest.json \
  --checkpoint-dir ../checkpoints/fine_tuning

# Stage 4: Multi-scene generalization (10-15 hours)
python production_training.py \
  --stage generalization \
  --episodes 200 \
  --scenes FloorPlan10,FloorPlan15,FloorPlan20 \
  --resume ../checkpoints/fine_tuning/latest.json \
  --checkpoint-dir ../checkpoints/production
```

### **Parallel Multi-GPU Training**
```bash
# Terminal 1: Scene 1
CUDA_VISIBLE_DEVICES=0 python production_training.py --scene FloorPlan1 --gpu 0 &

# Terminal 2: Scene 2
CUDA_VISIBLE_DEVICES=1 python production_training.py --scene FloorPlan2 --gpu 1 &

# Terminal 3: Scene 3
CUDA_VISIBLE_DEVICES=2 python production_training.py --scene FloorPlan5 --gpu 2 &

# Wait for all to complete
wait
```

---

## **4. MONITORING AND EVALUATION**

### **Real-time Training Monitoring**
```bash
# Monitor training progress with plots
python scripts/monitor_training.py \
  --checkpoint-dir checkpoints/production \
  --refresh 30 \
  --plots

# Generate training report only
python scripts/monitor_training.py \
  --checkpoint-dir checkpoints/production \
  --report-only
```

### **Quick Evaluation During Training**
```bash
# Quick performance check
python scripts/quick_eval.py \
  --checkpoint checkpoints/latest.json \
  --tasks 10 \
  --scene FloorPlan1 \
  --output results/quick_eval.json

# Verbose output with detailed results
python scripts/quick_eval.py \
  --tasks 20 \
  --scene FloorPlan1 \
  --verbose
```

### **Comprehensive Evaluation with Baselines**
```bash
# Full evaluation with baseline comparisons
python scripts/full_evaluation.py \
  --checkpoint checkpoints/production/final.json \
  --scenes FloorPlan1,FloorPlan2,FloorPlan5 \
  --tasks 100 \
  --baselines map_free,fixed_labels,oracle \
  --output results/comprehensive_evaluation.json

# Quick baseline comparison
python scripts/full_evaluation.py \
  --scenes FloorPlan1 \
  --tasks 20 \
  --baselines map_free,oracle \
  --output results/quick_comparison.json
```

### **Semantic Field Visualization**
```bash
# Visualize single semantic query
python scripts/visualize_field.py \
  --mode single \
  --query "dining table with chairs" \
  --scene FloorPlan1 \
  --output visualizations/table_field.png

# Compare multiple queries
python scripts/visualize_field.py \
  --mode multiple \
  --queries "dining table,comfortable sofa,kitchen counter,bed" \
  --scene FloorPlan1 \
  --output visualizations/multi_query.png

# Visualize field evolution during training
python scripts/visualize_field.py \
  --mode evolution \
  --query "dining table with chairs" \
  --checkpoint-dir checkpoints/production \
  --output visualizations/field_evolution.png

# Create query comparison grid
python scripts/visualize_field.py \
  --mode grid \
  --output visualizations/query_grid.png
```

---

## **5. RESUMING INTERRUPTED TRAINING**

### **Find and Resume from Checkpoints**
```bash
# List available checkpoints
ls -la checkpoints/*/latest.json

# Resume production training from checkpoint
python production_training.py \
  --resume checkpoints/object_learning/latest.json \
  --stage fine_tuning \
  --checkpoint-dir checkpoints/resumed

```

---

## **6. CONFIGURATION AND CUSTOMIZATION**

### **Custom Training Configuration**
```bash
# Create custom config
cat > configs/custom_config.yaml << EOF
clip_fields:
  batch_size: 24
  num_levels: 12
  resolution: 0.08
  learning_rate: 0.003

thor:
  max_steps_per_task: 400
  scenes: ["FloorPlan1", "FloorPlan7"]
  success_distance: 0.8

training:
  exploration_episodes: 100
  object_learning_episodes: 200
  validation_episodes: 50
EOF

# Use custom config
python rapid_training.py --config configs/custom_config.yaml --mode full
```

### **Performance Tuning**
```bash
# High-performance training (requires powerful GPU)
python production_training.py \
  --stage full \
  --scenes FloorPlan1 \
  --checkpoint-dir checkpoints/high_perf

# Memory-efficient training (for limited GPU memory)
export CLIP_BATCH_SIZE=8
export CLIP_RESOLUTION=0.15
python rapid_training.py --mode full --scene FloorPlan1
```

---

## **7. DEBUGGING AND TROUBLESHOOTING**

### **Check System Status**
```bash
# Verify server connection
python -c "
from bridge.communication_bridge import CLIPFieldsClient
client = CLIPFieldsClient()
try:
    status = client.get_field_status()
    print('✓ Server connected:', status)
except Exception as e:
    print('✗ Connection failed:', e)
"

# Check training progress
python scripts/monitor_training.py \
  --checkpoint-dir checkpoints/ \
  --report-only

# Test semantic queries
python examples/demo.py --demo-type query
```

### **Performance Debugging**
```bash
# Profile training performance
python -m cProfile -o profile_output.prof rapid_training.py --mode exploration --episodes 5

# Monitor GPU usage
nvidia-smi -l 1

# Check memory usage
python -c "
import torch
print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.memory_reserved()/1e9:.1f}GB')
"
