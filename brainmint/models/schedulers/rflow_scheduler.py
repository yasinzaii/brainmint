import torch.nn as nn
from monai.networks.schedulers.rectified_flow import RFlowScheduler as _RFlow

class RFlowSchedulerModule(nn.Module):
    """
    Lightning-safe wrapper that *constructs* MONAI's RFlowScheduler internally.
    Accepts the same kwargs as monai.networks.schedulers.rectified_flow.RFlowScheduler.
    All unknown attribute/method accesses are proxied to the inner scheduler.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # IMPORTANT: bypass Module.__setattr__ so the broken nn.Module inside DOES NOT get registered
        self.__dict__['_sched'] = _RFlow(**kwargs)

    def __getattr__(self, name):
        # Only called if normal lookup fails; forward to inner scheduler.
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self._sched, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._sched)})"