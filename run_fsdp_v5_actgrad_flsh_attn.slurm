#!/bin/bash
#SBATCH --job-name=informer-fsdp
#SBATCH --mail-user=shivansh@prl.res.in
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:2                      # 2 A100 GPUs per node
#SBATCH --nodes=4                         # 4 compute nodes
#SBATCH --ntasks-per-node=1               # 1 task per node (torchrun spawns GPU processes)
#SBATCH --cpus-per-task=4                 # CPUs per task (WORKS ON YOUR HPC)
#SBATCH --mem=32G                         # Memory per node (WORKS ON YOUR HPC)
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/informer_fsdp_%j.log
#SBATCH --error=logs/informer_fsdp_%j.error

# ============================================================================
# INFORMER FSDP TRAINING - WITH ACTIVATION CHECKPOINTING & GRADIENT ACCUMULATION
# Settings optimized for your HPC cluster
# ============================================================================

echo "=========================================================================="
echo "      INFORMER FSDP TRAINING - 4 Nodes × 2 A100 GPUs"
echo "      WITH ACTIVATION CHECKPOINTING & GRADIENT ACCUMULATION"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started on: $(hostname)"
echo "Start time: $(date)"
echo "=========================================================================="


# ============================================================================
# GPU and Node Detection
# ============================================================================
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_JOB_GPUS" ]; then
    NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

NUM_NODES=${SLURM_JOB_NUM_NODES:-4}
WORLD_SIZE=$((NUM_GPUS * NUM_NODES))



echo "Cluster Configuration:"
echo "  - Nodes: $NUM_NODES"
echo "  - GPUs per node: $NUM_GPUS (NVIDIA A100 80GB)"
echo "  - Total GPUs (world size): $WORLD_SIZE"
echo "  - CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  - Memory per node: ${SLURM_MEM_PER_NODE}MB (32GB)"
echo "  - Launch method: torchrun"
echo "=========================================================================="

echo "Allocated nodes:"
scontrol show hostname $SLURM_NODELIST | nl
echo "=========================================================================="

# ============================================================================
# Environment Variables for Distributed Training
# ============================================================================
# Master node configuration
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$WORLD_SIZE
export NNODES=$NUM_NODES

# NCCL Configuration for A100 GPUs
export NCCL_DEBUG=WARN                    # WARN for production (less verbose)
export NCCL_IB_DISABLE=0                  # Enable InfiniBand
export NCCL_IB_GID_INDEX=3                # InfiniBand GID index
export NCCL_SOCKET_IFNAME=^docker0,lo     # Exclude docker and loopback
export NCCL_NET_GDR_LEVEL=5               # GPU Direct RDMA for A100
export NCCL_P2P_LEVEL=NVL                 # NVLink for peer-to-peer
export NCCL_ALGO=Ring                     # Ring algorithm
export NCCL_MIN_NCHANNELS=4               # Minimum channels

# PyTorch distributed backend
export TORCH_DISTRIBUTED_DEBUG=OFF        # OFF for production (use DETAIL for debugging)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# CUDA Memory Management - CRITICAL for FSDP
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# CPU Threading - optimal for your configuration
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NUM_GPUS))  # 2 threads per GPU
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# Timeouts
export NCCL_TIMEOUT=3600                  # 1 hour
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300

echo "Environment Configuration:"
echo "  - MASTER_ADDR: $MASTER_ADDR"
echo "  - MASTER_PORT: $MASTER_PORT"
echo "  - WORLD_SIZE: $WORLD_SIZE"
echo "  - OMP_NUM_THREADS: $OMP_NUM_THREADS (per GPU)"
echo "  - NCCL_DEBUG: $NCCL_DEBUG"
echo "=========================================================================="

# ============================================================================
# Create Log Directory
# ============================================================================
mkdir -p logs
echo "✅ Log directory created/verified: logs/"
echo "=========================================================================="

# ============================================================================
# Navigate to Working Directory
# ============================================================================
WORK_DIR="/home/akashg/time-series/python-files"
cd $WORK_DIR || {
    echo "❌ ERROR: Cannot change to directory: $WORK_DIR"
    exit 1
}

echo "Working directory: $(pwd)"
echo "=========================================================================="

# ============================================================================
# Conda Environment Setup
# ============================================================================
echo "Setting up Python environment..."
source /home/akashg/miniconda3/etc/profile.d/conda.sh

echo "Activating informerworking environment..."
conda activate informerworking

# Verify Python environment
echo "Python Configuration:"
echo "  - Conda environment: $CONDA_DEFAULT_ENV"
echo "  - Python: $(which python)"
echo "  - Python version: $(python --version 2>&1)"
echo "  - PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  - CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "  - cuDNN version: $(python -c 'import torch; print(torch.backends.cudnn.version())')"
echo "  - GPUs visible: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  - torchrun: $(which torchrun)"
echo "=========================================================================="

# ============================================================================
# GPU Information (on master node)
# ============================================================================
echo "GPU Hardware Information on $(hostname):"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap --format=csv
echo "=========================================================================="

# Display GPU topology (helpful for debugging)
echo "GPU Topology on $(hostname):"
nvidia-smi topo -m
echo "=========================================================================="

# ============================================================================
# Test Inter-Node Connectivity
# ============================================================================
echo "Testing inter-node connectivity..."
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --ntasks-per-node=1 \
     bash -c "echo \"✅ Node \$(hostname) reachable (Node rank: \$SLURM_NODEID)\""
echo "=========================================================================="

# ============================================================================
# Verify Required Files Exist
# ============================================================================
echo "Verifying required files..."
REQUIRED_FILES=(
    "main_informer_auto_gpu_actgrad_flsh_attn.py"
    "Zhou_fsdp_actgrad_flsh_attn/exp/exp_informer.py"
    "Zhou_fsdp_actgrad_flsh_attn/exp/exp_basic.py"
    "Zhou_fsdp_actgrad_flsh_attn/data/data_loader.py"
    "Zhou_fsdp_actgrad_flsh_attn/utils/tools.py"
)

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ MISSING: $file"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    echo ""
    echo "❌ ERROR: Required files are missing!"
    echo "Please ensure all fixed Python files are installed."
    exit 1
fi
echo "=========================================================================="

# ============================================================================
# Build Command Line Arguments

# ============================================================================
# Launch FSDP Distributed Training with torchrun
# ============================================================================
echo "Launching FSDP Training with torchrun..."
echo "Configuration Summary:"
echo "  - Nodes: $NUM_NODES"
echo "  - GPUs per node: $NUM_GPUS"
echo "  - Total GPUs: $WORLD_SIZE"
echo "  - Launcher: torchrun --nproc_per_node=$NUM_GPUS"
echo "  - Master: $MASTER_ADDR:$MASTER_PORT"
echo "  - Training script: main_informer_auto_gpu_actgrad.py"
echo "  - FSDP: Enabled (model sharded across all $WORLD_SIZE GPUs)"
echo "=========================================================================="
echo "Training start: $(date)"
echo "=========================================================================="

# Use srun to launch torchrun on each node
# Each node runs one torchrun instance, which spawns NUM_GPUS processes (1 per GPU)
srun --nodes=$NUM_NODES \
     --ntasks=$NUM_NODES \
     --ntasks-per-node=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     --kill-on-bad-exit=1 \
     bash -c "
         # Set node rank for this node
         export NODE_RANK=\$SLURM_NODEID
         
         echo \"========================================================================\"
         echo \"Node \$(hostname) (rank \$NODE_RANK): Launching torchrun with $NUM_GPUS processes\"
         echo \"========================================================================\"
         
         # Show GPU visibility on this node
         echo \"GPUs visible on \$(hostname):\"
         nvidia-smi --query-gpu=index,name --format=csv,noheader
         echo \"\"
         
         # Launch torchrun - it will spawn $NUM_GPUS processes (one per GPU)
         torchrun \
             --nnodes=$NUM_NODES \
             --nproc_per_node=$NUM_GPUS \
             --node_rank=\$NODE_RANK \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             --rdzv_backend=c10d \
             --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
              main_informer_auto_gpu_actgrad_flsh_attn.py
     "

# Capture exit code
EXIT_CODE=$?

# ============================================================================
# Job Summary
# ============================================================================
echo "=========================================================================="
echo "                          JOB SUMMARY"
echo "=========================================================================="
echo "Training completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Training utilized:"
    echo "  - $NUM_NODES compute nodes"
    echo "  - $WORLD_SIZE NVIDIA A100 80GB GPUs"
    echo "  - FSDP distributed training (model sharded)"
    echo "  - PyTorch native torchrun launcher"
    echo ""
    echo "Checkpoints saved in: ./checkpoints/"
    echo "Results saved in: ./results/"
    echo "Logs available in: logs/informer_fsdp_${SLURM_JOB_ID}.log"
else
    echo "❌ TRAINING FAILED (Exit code: $EXIT_CODE)"
    echo ""
    echo "Common issues to check:"
    echo "  1. CUDA OOM:"
    echo "     → Reduce BATCH_SIZE (currently: $BATCH_SIZE)"
    echo "     → Reduce sequence lengths (seq_len, label_len, pred_len)"
    echo "     → Set FSDP_CPU_OFFLOAD=true"
    echo "     → Activation checkpointing is: $FSDP_ACTIVATION_CHECKPOINTING"
    echo ""
    echo "  2. NCCL Errors:"
    echo "     → Check network connectivity between nodes"
    echo "     → Verify InfiniBand is working"
    echo "     → Set NCCL_DEBUG=INFO and retry"
    echo ""
    echo "  3. File Not Found:"
    echo "     → Verify all fixed Python files are installed"
    echo "     → Check working directory: $WORK_DIR"
    echo ""
    echo "  4. Import Errors:"
    echo "     → Verify conda environment: informerworking"
    echo "     → Check PyTorch and CUDA versions"
    echo ""
    echo "Detailed logs:"
    echo "  - Output: logs/informer_fsdp_${SLURM_JOB_ID}.log"
    echo "  - Errors: logs/informer_fsdp_${SLURM_JOB_ID}.error"
fi

echo "=========================================================================="
echo "Job end time: $(date)"
echo "=========================================================================="

exit $EXIT_CODE
