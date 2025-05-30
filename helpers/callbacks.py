from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import Callback
from time import time

from lightning.pytorch.utilities import grad_norm
from torch.nn.utils import clip_grad_norm_

class TimingCallback(Callback):
    """
    Class that computes train and validation epoch time.
    """
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_start_time = time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.log("epoch_time", time() - self.epoch_start_time, sync_dist=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.log("validation_epoch_time", time() - self.validation_epoch_start_time, sync_dist=True)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.validation_epoch_start_time = time()


def get_callbacks(cfg) -> list:
 
    checkpoint_cb = ModelCheckpoint(
        every_n_epochs=cfg.trainer.checkpoint_every_n_epochs,
    ) # Save only the last checkpoint

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    grad_norm_cb = GradNormWithClip(cfg)

    callbacks = [checkpoint_cb, lr_monitor, grad_norm_cb]

    return callbacks


class GradNormWithClip(Callback):
    def __init__(self, cfg):
        """
        Args:
            max_norm: the clipping threshold (same semantics as `gradient_clip_val`)
            norm_type: p-norm degree
        """
        self.max_norm = cfg.optim.max_grad_norm
        self.norm_type = cfg.optim.grad_norm_type
        self.bsz = cfg.data.batch_size

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # 1) Pre-clip norm from Lightning util
        norms = grad_norm(pl_module, norm_type=self.norm_type)
        pre = norms[f"grad_{self.norm_type}_norm_total"]

        # 2) Do the clip ourselves (in-place on p.grad)
        clip_grad_norm_(pl_module.parameters(), self.max_norm, self.norm_type)

        # 3) Compute post-clip norm from the same util
        norms_after = grad_norm(pl_module, norm_type=self.norm_type)
        post = norms_after[f"grad_{self.norm_type}_norm_total"]

        # 4) Log both
        pl_module.log("optim/grad_norm_preclip", pre, on_epoch=True, on_step=False, batch_size=self.bsz)
        pl_module.log("optim/grad_norm_postclip", post, on_epoch=True, on_step=False, batch_size=self.bsz)