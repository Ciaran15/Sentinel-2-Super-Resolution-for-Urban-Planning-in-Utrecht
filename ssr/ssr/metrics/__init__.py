from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

try:
    from .clipscore import calculate_clipscore
except ImportError:
    calculate_clipscore = None

try:
    from .cpsnr import calculate_cpsnr
except ImportError:
    calculate_cpsnr = None

try:
    from .lpips import calculate_lpips
except ImportError:
    calculate_lpips = None

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_clipscore', 'calculate_cpsnr', 'calculate_lpips']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
