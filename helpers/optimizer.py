import numpy as np

class CosineWithWarmup:
    def __init__(self, warmup_steps=0, warmup_factor=0.1, total_steps=None) -> None:
        assert total_steps is not None
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.total_steps = total_steps

    def __call__(self, step):
        factor = 0.5 * (1.0 + np.cos(np.pi * step / self.total_steps))
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            warmup = self.warmup_factor * (1.0 - alpha) + alpha
            factor *= warmup
        return factor






