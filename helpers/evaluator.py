import torch

from torchmetrics.image.fid import FrechetInceptionDistance
#from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


class Evaluator:
    def __init__(self, device='cuda:0') -> None:
        self.device = device
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device) # [0,1] range
        #self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=False).to(device) # [0,1] range
        self.kid = KernelInceptionDistance(normalize=True,subset_size=4).to(device) # [0,1] range
        self.metrics = {
            'fid': self.fid, # ↓
            #'ssim': self.ssim,
            'psnr': self.psnr, # ↑
            'lpips': self.lpips, # ↓
            'kid': self.kid # ↓
        }
        self.reset()
    
    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self.count = 0
        self.results = {
            'fid': [],
            #'ssim': [],
            'psnr': [],
            'lpips': [],
            'kid_mean': [],
            'kid_std': []
        }

    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
        """
        Update the metrics with new images.
        Args:
            real_images (torch.Tensor): Real images.
            fake_images (torch.Tensor): Fake images.
        """
        for metric in self.metrics:
            if metric in ['fid','kid']:
                self.metrics[metric].update(real_images, real=True)
                self.metrics[metric].update(fake_images, real=False)
            else:
                self.metrics[metric].update(real_images,fake_images)
        self.count += 1
    
    def compute(self) -> None:
        self.results['fid'].append(self.fid.compute().item())
        #self.results['ssim'].append(self.ssim.compute().item())
        self.results['psnr'].append(self.psnr.compute().item())
        self.results['lpips'].append(self.lpips.compute().item())
        kid_mean,kid_std = self.kid.compute()
        self.results['kid_mean'].append(kid_mean.item())
        self.results['kid_std'].append(kid_std.item())

    def get_results(self) -> dict:
        """
        Get the results of the metrics.
        Returns:
            dict: Dictionary with the results of the metrics.
        """
        return {
            'fid': sum(self.results['fid']) / self.count,
            #'ssim': sum(self.results['ssim']) / self.count,
            'psnr': sum(self.results['psnr']) / self.count,
            'lpips': sum(self.results['lpips']) / self.count,
            'kid_mean': sum(self.results['kid_mean']) / self.count,
            'kid_std': sum(self.results['kid_std']) / self.count
        }
    
    def log_results(self) -> None:
        """
        Log the results of the metrics.
        Args:
            epoch (int): Current epoch.
        """
        results = self.get_results()
        # if logger.__class__.__name__ == "WandbLogger":
        #     for metric, value in results.items():
        #         logger.experiment.log({f"{metric}/epoch": value}, step=global_step)
        self.reset()
        return results


if __name__ == "__main__":
    evaluator = Evaluator()
    # Example usage
    real_images = torch.rand(8, 3, 256, 256).to(evaluator.device)
    fake_images = torch.rand(8, 3, 256, 256).to(evaluator.device)

    evaluator.update(real_images, fake_images)
    evaluator.compute()
    results = evaluator.get_results()
    print(results)

    evaluator.update(real_images, real_images)
    evaluator.compute()
    results = evaluator.get_results()
    print(results)

