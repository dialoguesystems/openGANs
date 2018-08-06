"""Module defining various utilities."""
from ogan.utils.misc import aeq, use_gpu
from ogan.utils.report_manager import ReportMgr, build_report_manager
from ogan.utils.statistics import Statistics
from ogan.utils.optimizers import build_optim, MultipleOptimizer, \
    Optimizer

# import ogan.utils.loss

__all__ = ["aeq", "use_gpu", "ReportMgr",
           "build_report_manager", "Statistics",
           "build_optim", "MultipleOptimizer", "Optimizer"]