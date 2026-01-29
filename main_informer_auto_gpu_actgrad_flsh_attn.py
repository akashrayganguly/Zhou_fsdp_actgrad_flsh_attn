#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Informer Training Script with FSDP Support and Automatic GPU Detection

FIXES FROM ORIGINAL:
1. Added gradient_accumulation_steps argument
2. Added max_grad_norm argument for gradient clipping
3. Better configuration printing

MODIFICATIONS:
- batch_size increased from 1 to 2
- use_amp enabled (True) for mixed precision training
"""

import sys
import os
import torch
import torch.distributed as dist
from datetime import timedelta



# Add project directory to path
if 'Zhou_fsdp_actgrad_flsh_attn' not in sys.path:
    sys.path.append('Zhou_fsdp_actgrad_flsh_attn')


from utils.tools import dotdict
from exp.exp_informer import Exp_Informer



def get_available_gpus():
    """Get the number of available GPUs on this node"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def setup_distributed():
    """Initialize distributed training environment with automatic GPU detection"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
    else:
        num_gpus = get_available_gpus()
        
        if num_gpus > 1:
            print(f"⚠️  WARNING: {num_gpus} GPUs detected but not using distributed training!")
            print(f"   To use all GPUs, run with: torchrun --nproc_per_node={num_gpus} {sys.argv[0]}")
        
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
    
    return rank, world_size, local_rank


def init_distributed_mode(args):
    """Initialize distributed training with automatic GPU detection"""
    if args.use_fsdp:
        rank, world_size, local_rank = setup_distributed()
        
        num_gpus = get_available_gpus()
        
        if torch.cuda.is_available():
            if local_rank >= num_gpus:
                raise RuntimeError(
                    f"Local rank {local_rank} is >= number of GPUs {num_gpus}."
                )
            
            if rank == 0:
                print(f"\n{'=' * 80}")
                print(f"GPU Configuration:")
                print(f"  - GPUs per node: {num_gpus}")
                print(f"  - Total world size: {world_size}")
                print(f"  - Processes per node: {world_size if world_size <= num_gpus else num_gpus}")

                if world_size > num_gpus:
                    num_nodes = world_size // num_gpus
                    print(f"  - Number of nodes: {num_nodes}")

                print(f"  - Assignment: 1 process per GPU")
                print(f"{'=' * 80}\n")
        
        args.global_rank = rank
        args.world_size = world_size
        args.local_rank = local_rank
        args.num_gpus_per_node = num_gpus
        
        if not dist.is_initialized() and world_size > 1:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            
            if rank == 0:
                print(f"Initializing distributed process group with {backend} backend...")
            
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=30)
            )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = torch.device(f'cuda:{local_rank}')
            args.gpu = local_rank

            # Verify GPU assignment
            if rank == 0:
                print(f"Process-to-GPU Assignment:")

            # Each rank prints its assignment (synchronized)
            for i in range(world_size):
                if rank == i:
                    gpu_name = torch.cuda.get_device_name(local_rank)
                    print(f"  Rank {rank:2d} (local_rank {local_rank}) → GPU {local_rank}: {gpu_name}")
                if dist.is_initialized():
                    dist.barrier()
        else:
            args.device = torch.device('cpu')
            args.gpu = None
            if rank == 0:
                print("No CUDA devices available. Running on CPU.")

        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            print(f"Distributed training initialized: world_size={world_size}")
    else:
        args.global_rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.num_gpus_per_node = get_available_gpus()


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_args():
    """Create and configure arguments"""
    args = dotdict()

    # ============================================================================
    # Model configuration
    # ============================================================================
    args.model = 'informer'

    # ============================================================================
    # Data configuration
    # ============================================================================
    args.data = 'custom'
    args.root_path = './ETDataset/ETT-small/'
    args.data_path = 'nc_by_meff_multiple_mtau_ECL.csv'
    args.features = 'M'
    args.target = 'data9'
    args.freq = 'h'
    args.checkpoints = './checkpoints'
    args.cols = None
    args.inverse = False

    # ============================================================================
    # Sequence lengths
    # ============================================================================
    args.seq_len = 96*321
    args.label_len = 48*321
    args.pred_len = 24*321

    # ============================================================================
    # Model parameters
    # ============================================================================
    args.enc_in = 9
    args.dec_in = 9
    args.c_out = 9
    args.factor = 5
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.s_layers = [3, 2, 1]
    args.d_ff = 2048
    args.dropout = 0.05
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.distil = True
    args.output_attention = False
    args.mix = True
    args.padding = 0

    # ============================================================================
    # Training parameters
    # ============================================================================
    # MODIFIED: Increased batch_size from 1 to 2
    args.batch_size = 2
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    # MODIFIED: Enabled mixed precision training (was False)
    args.use_amp = True
    args.train_epochs = 10
    args.patience = 3
    
    # ============================================================================
    # NEW: Gradient Accumulation
    # ============================================================================
    args.gradient_accumulation_steps = 1  # Effective batch = batch_size × this × world_size
    args.max_grad_norm = 1.0  # Gradient clipping (0 to disable)
    
    # ============================================================================
    # Experiment settings
    # ============================================================================
    args.num_workers = 4
    args.itr = 1
    args.des = 'fsdp_exp'
    args.seed = 2021

    # ============================================================================
    # GPU settings - AUTOMATIC DETECTION
    # ============================================================================
    args.use_gpu = True if torch.cuda.is_available() else False

    # Enable FSDP for distributed training
    args.use_fsdp = True
    args.use_multi_gpu = False  # Don't use DataParallel

    # These are legacy parameters (not used in FSDP mode)
    args.gpu = 0
    args.devices = 'auto'
    args.device_ids = None
    
    # ============================================================================
    # FSDP configuration
    # ============================================================================
    args.fsdp_sharding_strategy = 'FULL_SHARD'
    args.fsdp_auto_wrap_min_params = 1e6
    args.fsdp_backward_prefetch = 'BACKWARD_PRE'
    args.fsdp_cpu_offload = False
    args.fsdp_activation_checkpointing = True  # Set to True to enable

    return args


def setup_gpu(args):
    """
    Setup GPU configuration for non-FSDP mode
    For FSDP, device is set in init_distributed_mode()
    """
    if not args.use_fsdp:
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu:
            num_gpus = get_available_gpus()
            print(f"Number of GPUs available: {num_gpus}")

            if args.use_multi_gpu and num_gpus > 1:
                print(f"⚠️  DataParallel mode with {num_gpus} GPUs")
                print(f"   Consider using FSDP for better performance and scalability!")

                # Auto-detect all available GPUs
                args.device_ids = list(range(num_gpus))
                args.gpu = 0
                print(f"   Using GPUs: {args.device_ids}")
            else:
                args.gpu = 0
                args.device_ids = [0]
                args.use_multi_gpu = False

            torch.cuda.set_device(args.gpu)
            print(f"Using GPU: {args.gpu}")
        else:
            print("Using CPU")
    else:
        if args.global_rank == 0:
            print(f"FSDP mode: Device management handled automatically")


def setup_data_parser(args):
    """Configure data parser"""
    data_parser = {
        'custom': {
            'data': 'nc_by_meff_multiple_mtau_ECL.csv',
            'T': 'data9',
            'M': [9, 9, 9],
            'S': [1, 1, 1],
            'MS': [9, 9, 1]
        },
        'ETTh1': {
            'data': 'ETTh1.csv',
            'T': 'OT',
            'M': [7, 7, 7],
            'S': [1, 1, 1],
            'MS': [7, 7, 1]
        },
        'ETTh2': {
            'data': 'ETTh2.csv',
            'T': 'OT',
            'M': [7, 7, 7],
            'S': [1, 1, 1],
            'MS': [7, 7, 1]
        },
        'ETTm1': {
            'data': 'ETTm1.csv',
            'T': 'OT',
            'M': [7, 7, 7],
            'S': [1, 1, 1],
            'MS': [7, 7, 1]
        },
        'ETTm2': {
            'data': 'ETTm2.csv',
            'T': 'OT',
            'M': [7, 7, 7],
            'S': [1, 1, 1],
            'MS': [7, 7, 1]
        },
        'WTH': {
            'data': 'WTH.csv',
            'T': 'WetBulbCelsius',
            'M': [12, 12, 12],
            'S': [1, 1, 1],
            'MS': [12, 12, 1]
        },
        'ECL': {
            'data': 'ECL.csv',
            'T': 'MT_320',
            'M': [321, 321, 321],
            'S': [1, 1, 1],
            'MS': [321, 321, 1]
        },
        'Solar': {
            'data': 'solar_AL.csv',
            'T': 'POWER_136',
            'M': [137, 137, 137],
            'S': [1, 1, 1],
            'MS': [137, 137, 1]
        },
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]


def print_args(args):
    """Print arguments (only from rank 0 in distributed mode)"""
    should_print = not args.use_fsdp or args.global_rank == 0
    
    if should_print:
        print("\n" + "=" * 80)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 80)
        print(f"Model: {args.model}")
        print(f"Data: {args.data} ({args.data_path})")
        print(f"Features: {args.features}")
        print(f"Sequence lengths: seq={args.seq_len}, label={args.label_len}, pred={args.pred_len}")
        print(f"Model params: d_model={args.d_model}, n_heads={args.n_heads}, layers=E{args.e_layers}/D{args.d_layers}")
        print(f"Training: epochs={args.train_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
        
        # Mixed precision info
        print(f"\nMixed Precision Training:")
        print(f"  - Enabled: {args.use_amp}")
        if args.use_amp:
            print(f"  - Using FP16 with GradScaler for automatic loss scaling")
        
        # Gradient accumulation info
        effective_batch = args.batch_size * args.gradient_accumulation_steps * args.world_size
        print(f"\nGradient Accumulation:")
        print(f"  - Steps: {args.gradient_accumulation_steps}")
        print(f"  - Effective batch size: {effective_batch}")
        print(f"  - Max grad norm: {args.max_grad_norm}")
        
        print(f"\nDevice Configuration:")
        if args.use_fsdp:
            print(f"  Mode: FSDP (Fully Sharded Data Parallel)")
            print(f"  GPUs per node: {args.num_gpus_per_node}")
            print(f"  World size: {args.world_size}")
            print(f"  Sharding strategy: {args.fsdp_sharding_strategy}")
            print(f"  CPU offload: {args.fsdp_cpu_offload}")
            print(f"  Activation checkpointing: {args.fsdp_activation_checkpointing}")
        elif args.use_multi_gpu:
            print(f"  Mode: DataParallel")
            print(f"  Devices: {args.device_ids}")
        else:
            print(f"  Mode: Single GPU/CPU")
            print(f"  Device: GPU {args.gpu}" if args.use_gpu else "CPU")
        
        print("=" * 80 + "\n")


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_usage_instructions():
    """Print usage instructions for different scenarios"""
    num_gpus = get_available_gpus()

    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)

    if num_gpus == 0:
        print("No GPUs detected. Running on CPU.")
        print("\nCommand: python main_informer.py")

    elif num_gpus == 1:
        print(f"1 GPU detected.")
        print("\nSingle GPU training:")
        print("  python main_informer.py")

    else:
        print(f"{num_gpus} GPUs detected on this node.")
        print(f"\n✅ RECOMMENDED: Use all {num_gpus} GPUs with FSDP:")
        print(f"  torchrun --nproc_per_node={num_gpus} main_informer.py")
        print(f"\nThis will automatically:")
        print(f"  - Spawn {num_gpus} processes")
        print(f"  - Assign 1 process per GPU")
        print(f"  - Use FSDP for distributed training")

        print(f"\nAlternative: Single GPU training:")
        print(f"  python main_informer.py")
        print(f"  (Will only use GPU 0)")

    print("=" * 80 + "\n")


def main():
    """Main training function"""
    try:
        # Setup
        args = create_args()
        
        # Print usage instructions if running in single-process mode with multiple GPUs
        if 'RANK' not in os.environ and get_available_gpus() > 1 and args.use_fsdp:
            print_usage_instructions()

        # Initialize distributed training if FSDP is enabled
        if args.use_fsdp:
            init_distributed_mode(args)
        else:
            args.global_rank = 0
            args.world_size = 1
            args.local_rank = 0
            args.num_gpus_per_node = get_available_gpus()
        
        # Print header (only rank 0)
        if args.global_rank == 0:
            print("\n" + "=" * 80)
            print("INFORMER TRAINING WITH FSDP - AUTO GPU DETECTION")
            print("=" * 80)
        
        # Setup GPU/device
        setup_gpu(args)
        
        # Setup data parser
        setup_data_parser(args)
        
        # Additional configuration
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]
        
        # Set random seed
        set_seed(args.seed + args.global_rank)
        
        # Print configuration
        print_args(args)
        
        # Training Loop
        Exp = Exp_Informer

        for ii in range(args.itr):
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii
            )

            if args.global_rank == 0:
                print(f"\n{'=' * 80}")
                print(f"ITERATION {ii + 1}/{args.itr}")
                print(f"Setting: {setting}")
                print(f"{'=' * 80}\n")

            # Create experiment
            exp = Exp(args)

            # Training
            if args.global_rank == 0:
                print(f'\n>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            # Testing
            if args.global_rank == 0:
                print(f'\n>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if args.use_fsdp and dist.is_initialized():
                dist.barrier()

        # Done
        if args.global_rank == 0:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 80 + "\n")
        
        # Cleanup
        if args.use_fsdp:
            cleanup_distributed()

    except Exception as e:
        if 'args' in locals() and hasattr(args, 'use_fsdp') and args.use_fsdp:
            if hasattr(args, 'global_rank') and args.global_rank == 0:
                print(f"\nERROR: {str(e)}")
                import traceback
                traceback.print_exc()
            cleanup_distributed()
        else:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()

        raise


if __name__ == "__main__":
    main()
