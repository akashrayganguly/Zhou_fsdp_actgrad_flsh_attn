"""
exp_informer.py - Complete Implementation with FSDP Support

FIXES FROM ORIGINAL:
1. Activation checkpointing uses CORRECT layer classes (EncoderLayer, DecoderLayer)
2. Activation checkpointing applied BEFORE FSDP wrapping (not after)
3. Added gradient accumulation support
4. Added use_orig_params=True for better optimizer compatibility
5. Using transformer_auto_wrap_policy for better layer-wise sharding
6. Fixed loss accumulation to stay on GPU (minimize transfers)

MIXED PRECISION AUTO-DETECTION:
- Automatically detects if GPU supports BF16 (Ampere+: A100, RTX 30xx, etc.)
- BF16 supported: Uses BF16, NO GradScaler needed
- BF16 NOT supported: Uses FP16 WITH GradScaler (for V100, RTX 20xx, etc.)
- Fallback to FP32 if neither works
"""

import os
import time
import warnings
import numpy as np
import functools

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

# Activation checkpointing imports
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Mixed precision imports
from torch.cuda.amp import GradScaler, autocast

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

# =========================================================================
# CRITICAL: Import the ACTUAL layer classes used in the Informer model
# =========================================================================
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')


def get_amp_config():
    """
    Auto-detect the best mixed precision configuration for the current GPU.
    
    Returns:
        dict with keys:
            - 'supported': bool - whether AMP is supported
            - 'dtype': torch.dtype - the dtype to use (bfloat16, float16, or float32)
            - 'use_scaler': bool - whether GradScaler is needed
            - 'name': str - human readable name
    """
    if not torch.cuda.is_available():
        return {
            'supported': False,
            'dtype': torch.float32,
            'use_scaler': False,
            'name': 'FP32 (CPU/No CUDA)'
        }
    
    # Check for BF16 support (Ampere and newer: A100, RTX 30xx, H100, etc.)
    # torch.cuda.is_bf16_supported() checks compute capability >= 8.0
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        return {
            'supported': True,
            'dtype': torch.bfloat16,
            'use_scaler': False,  # BF16 doesn't need GradScaler
            'name': 'BFloat16 (Ampere+)'
        }
    
    # Check compute capability for FP16 (most CUDA GPUs support this)
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    compute_cap = capability[0] * 10 + capability[1]
    
    # FP16 with Tensor Cores requires compute capability >= 7.0 (Volta+)
    if compute_cap >= 70:
        return {
            'supported': True,
            'dtype': torch.float16,
            'use_scaler': True,  # FP16 NEEDS GradScaler to prevent overflow
            'name': 'Float16 (Volta+)'
        }
    
    # Fallback to FP32 for older GPUs
    return {
        'supported': False,
        'dtype': torch.float32,
        'use_scaler': False,
        'name': 'FP32 (Legacy GPU)'
    }


class Exp_Informer(Exp_Basic):
    """
    Informer Experiment Class with FSDP, Activation Checkpointing, and Gradient Accumulation
    """
    
    def __init__(self, args):
        # Set gradient accumulation steps before parent init
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        
        # Auto-detect AMP configuration
        self.amp_config = get_amp_config()
        
        super(Exp_Informer, self).__init__(args)
        
        if self._should_print():
            print(f"\n{'='*60}")
            print(f"Experiment Configuration:")
            print(f"  - FSDP Enabled: {getattr(args, 'use_fsdp', False)}")
            print(f"  - Activation Checkpointing: {getattr(args, 'fsdp_activation_checkpointing', False)}")
            print(f"  - Mixed Precision Requested: {getattr(args, 'use_amp', False)}")
            if getattr(args, 'use_amp', False):
                print(f"  - AMP Auto-Detected: {self.amp_config['name']}")
                print(f"  - AMP Dtype: {self.amp_config['dtype']}")
                print(f"  - GradScaler Required: {self.amp_config['use_scaler']}")
            print(f"  - Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
            effective_batch = args.batch_size * self.gradient_accumulation_steps * getattr(args, 'world_size', 1)
            print(f"  - Effective Batch Size: {effective_batch}")
            print(f"{'='*60}\n")

    def _build_model(self):
        """
        Build model with FSDP wrapping and optional activation checkpointing
        """
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }

        # Determine e_layers based on model type
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers

            # Build the base model
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        # Wrap model based on configuration
        if self.args.use_fsdp and torch.cuda.is_available():
            model = self._wrap_model_with_fsdp(model)
        elif self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _wrap_model_with_fsdp(self, model):
        """
        Wrap model with FSDP and optionally apply activation checkpointing
        
        Mixed Precision is auto-configured based on GPU capability:
        - Ampere+ (A100, RTX 30xx): BF16, no GradScaler
        - Volta/Turing (V100, RTX 20xx): FP16, with GradScaler
        - Older: FP32
        """
        if self._should_print():
            print(f"\n{'='*60}")
            print("Wrapping model with FSDP...")
            print(f"  - Sharding Strategy: {self.args.fsdp_sharding_strategy}")
            print(f"  - CPU Offload: {self.args.fsdp_cpu_offload}")
            print(f"  - Activation Checkpointing: {self.args.fsdp_activation_checkpointing}")
            if self.args.use_amp:
                print(f"  - Mixed Precision: {self.amp_config['name']}")
            else:
                print(f"  - Mixed Precision: Disabled (FP32)")
            print(f"{'='*60}\n")

        # =====================================================================
        # STEP 1: Apply activation checkpointing BEFORE FSDP wrapping
        # =====================================================================
        if self.args.fsdp_activation_checkpointing:
            self._apply_activation_checkpointing(model)

        # =====================================================================
        # STEP 2: Configure FSDP
        # =====================================================================
        
        sharding_strategy = getattr(ShardingStrategy, self.args.fsdp_sharding_strategy)
        backward_prefetch = getattr(BackwardPrefetch, self.args.fsdp_backward_prefetch)

        cpu_offload_policy = None
        if self.args.fsdp_cpu_offload:
            cpu_offload_policy = CPUOffload(offload_params=True)

        # =====================================================================
        # Mixed precision - AUTO-CONFIGURED based on GPU capability
        # =====================================================================
        mixed_precision_policy = None
        if self.args.use_amp and self.amp_config['supported']:
            amp_dtype = self.amp_config['dtype']
            mixed_precision_policy = MixedPrecision(
                param_dtype=amp_dtype,
                reduce_dtype=amp_dtype,
                buffer_dtype=amp_dtype,
            )
            if self._should_print():
                print(f"FSDP MixedPrecision Policy: {self.amp_config['name']}")

        # =====================================================================
        # Auto-wrap policy
        # =====================================================================
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={EncoderLayer, DecoderLayer},
        )

        # =====================================================================
        # STEP 3: Wrap with FSDP
        # =====================================================================
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload_policy,
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(),
            backward_prefetch=backward_prefetch,
            limit_all_gathers=True,
            use_orig_params=True,
        )

        if self._should_print():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model wrapped with FSDP:")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Trainable parameters: {trainable_params:,}")

        return model

    def _apply_activation_checkpointing(self, model):
        """
        Apply activation checkpointing to encoder and decoder layers

        CRITICAL FIX: Use the ACTUAL layer classes from models/encoder.py and models/decoder.py
        NOT nn.TransformerEncoderLayer/DecoderLayer!

        This wraps specific layers with checkpoint_wrapper, which:
        1. Does NOT store activations during forward pass
        2. Recomputes activations during backward pass
        3. Saves ~60% memory at cost of ~30% more compute
        """
        if self._should_print():
            print("Applying activation checkpointing...")
            print(f"  - Target layers: EncoderLayer, DecoderLayer")

        # =====================================================================
        # CRITICAL FIX: Check for YOUR model's layer classes, not PyTorch's!
        # Your model uses:
        #   - models.encoder.EncoderLayer (NOT nn.TransformerEncoderLayer)
        #   - models.decoder.DecoderLayer (NOT nn.TransformerDecoderLayer)
        # =====================================================================
        def check_fn(submodule):
            return isinstance(submodule, (EncoderLayer, DecoderLayer))

        # Use non-reentrant checkpointing (required for FSDP)
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        # Apply checkpointing to all matching layers
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn
        )

        if self._should_print():
            # Count checkpointed layers
            enc_count = sum(1 for m in model.modules() if isinstance(m, EncoderLayer))
            dec_count = sum(1 for m in model.modules() if isinstance(m, DecoderLayer))
            print(f"  - Checkpointed {enc_count} EncoderLayers, {dec_count} DecoderLayers")

    def _get_data(self, flag):
        """Get dataset and dataloader with FSDP support"""
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        # Determine batch size and shuffle behavior
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        # Create dataset
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )

        # Only rank 0 prints
        if self._should_print():
            print(f'{flag} dataset size: {len(data_set)}')

        # Create dataloader with FSDP support
        if hasattr(args, 'use_fsdp') and args.use_fsdp:
            # FSDP mode: Use DistributedSampler
            sampler = DistributedSampler(
                data_set,
                shuffle=shuffle_flag,
                seed=args.seed if hasattr(args, 'seed') else 2021,
                drop_last=drop_last
            )

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,  # CRITICAL: Must be False when using sampler!
                sampler=sampler,
                num_workers=args.num_workers,
                drop_last=drop_last,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False
            )
        else:
            # Standard mode: Regular DataLoader
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )

        return data_set, data_loader

    def _select_optimizer(self):
        """Select optimizer"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Select loss criterion"""
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """Validation loop with FSDP support"""
        self.model.eval()
        
        # Accumulate loss on GPU
        total_loss = torch.tensor(0.0, device=self.device)
        total_count = 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                # Always compute loss in FP32 for numerical stability
                loss = criterion(pred.float(), true.float())
                total_loss += loss.detach()
                total_count += 1

        # Average loss
        avg_loss = total_loss / total_count

        # Synchronize loss across all processes in FSDP
        if self.args.use_fsdp and dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

        self.model.train()
        return avg_loss.item()

    def train(self, setting):
        """
        Training loop with FSDP support and Gradient Accumulation
        
        BF16 IMPORTANT: No GradScaler needed!
        - BF16 has same exponent range as FP32
        - No overflow/underflow issues like FP16
        - Just use autocast + regular backward
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        
        # Only rank 0 creates directory
        if self._should_print():
            if not os.path.exists(path):
                os.makedirs(path)
        
        # Synchronize to ensure directory is created before all ranks continue
        if self.args.use_fsdp and dist.is_initialized():
            dist.barrier()

        time_now = time.time()
        train_steps = len(train_loader)
        
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            use_fsdp=self.args.use_fsdp if hasattr(self.args, 'use_fsdp') else False,
            global_rank=self.args.global_rank if hasattr(self.args, 'global_rank') else 0
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # =====================================================================
        # GradScaler - ONLY for FP16, NOT for BF16 or FP32
        # =====================================================================
        use_amp = self.args.use_amp and self.amp_config['supported']
        use_scaler = use_amp and self.amp_config['use_scaler']
        scaler = GradScaler() if use_scaler else None

        # Gradient accumulation
        grad_accum_steps = self.gradient_accumulation_steps

        if self._should_print():
            print(f"\n{'='*60}")
            print(f"Training Configuration:")
            print(f"  - Train steps per epoch: {train_steps}")
            print(f"  - Gradient accumulation steps: {grad_accum_steps}")
            print(f"  - Effective steps per epoch: {train_steps // grad_accum_steps}")
            print(f"  - Mixed Precision: {self.amp_config['name'] if use_amp else 'Disabled (FP32)'}")
            if use_amp:
                print(f"  - GradScaler: {'Enabled' if use_scaler else 'Not needed (BF16)'}")
            print(f"{'='*60}\n")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_sum = torch.tensor(0.0, device=self.device)
            train_loss_count = 0

            if self.args.use_fsdp and hasattr(train_loader, 'sampler'):
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                # Forward pass
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                
                # Compute loss in FP32 for numerical stability
                loss = criterion(pred.float(), true.float())
                loss_scaled = loss / grad_accum_steps

                with torch.no_grad():
                    train_loss_sum += loss.detach()
                    train_loss_count += 1

                # =====================================================================
                # Backward pass - different strategies based on precision
                # =====================================================================
                if use_scaler:
                    # FP16: Use GradScaler to prevent overflow
                    scaler.scale(loss_scaled).backward()
                else:
                    # BF16 or FP32: Direct backward (no scaler needed)
                    loss_scaled.backward()

                # Update weights every grad_accum_steps
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == train_steps:
                    if use_scaler:
                        # FP16 with GradScaler
                        if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm > 0:
                            scaler.unscale_(model_optim)
                            if self.args.use_fsdp:
                                self.model.clip_grad_norm_(self.args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        # BF16 or FP32: Direct step
                        if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm > 0:
                            if self.args.use_fsdp:
                                self.model.clip_grad_norm_(self.args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        model_optim.step()
                    
                    model_optim.zero_grad()

                if (i + 1) % 100 == 0 and self._should_print():
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            if self._should_print():
                print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")

            # Compute average training loss
            train_loss = (train_loss_sum / train_loss_count).item()
            
            # Synchronize training loss across all ranks
            if self.args.use_fsdp and dist.is_initialized():
                train_loss_tensor = torch.tensor(train_loss, device=self.device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = train_loss_tensor.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            if self._should_print():
                print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                      f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if self._should_print():
                    print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self._load_checkpoint(best_model_path)

        return self.model

    def test(self, setting):
        """Test loop with FSDP support"""
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                preds.append(pred.detach().cpu().float().numpy())
                trues.append(true.detach().cpu().float().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        if self._should_print():
            print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        if self._should_print():
            print('test shape:', preds.shape, trues.shape)

        # Result save (only from rank 0)
        folder_path = './results/' + setting + '/'
        
        if self._should_print():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        # Synchronize
        if self.args.use_fsdp and dist.is_initialized():
            dist.barrier()

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        if self._should_print():
            print(f'\n{"="*60}')
            print(f'Test Results:')
            print(f'  MSE: {mse:.6f}, MAE: {mae:.6f}')
            print(f'{"="*60}\n')

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        """Prediction with FSDP support"""
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self._load_checkpoint(best_model_path)

        self.model.eval()

        preds = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                preds.append(pred.detach().cpu().float().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # Result save (only from rank 0)
        folder_path = './results/' + setting + '/'
        
        if self._should_print():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        if self.args.use_fsdp and dist.is_initialized():
            dist.barrier()

        if self._should_print():
            np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        Process one batch of data
        
        Note: With FSDP + MixedPrecision policy, the model automatically
        runs in the configured precision (BF16/FP16). We don't need 
        explicit autocast here because FSDP's MixedPrecision handles it.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # =====================================================================
        # Encoder - Decoder
        # With FSDP + BF16 MixedPrecision policy, the model automatically
        # runs in BF16. We don't need explicit autocast here because
        # FSDP's MixedPrecision handles it.
        # =====================================================================
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def _load_checkpoint(self, path):
        """Load checkpoint with FSDP support"""
        if not os.path.exists(path):
            if self._should_print():
                print(f"Checkpoint not found at {path}")
            return
            
        if self._should_print():
            print(f"Loading checkpoint from {path}")
            
        if self.args.use_fsdp:
            # FSDP checkpoint loading
            with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,
            ):
                state_dict = torch.load(path, map_location='cpu')
                self.model.load_state_dict(state_dict)
        else:
            # Standard checkpoint loading
            state_dict = torch.load(path, map_location=self.device)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

    def _should_print(self):
        """Check if current process should print (only rank 0 in FSDP mode)"""
        if hasattr(self.args, 'use_fsdp') and self.args.use_fsdp:
            return self.args.global_rank == 0
        return True
