import importlib

import torch


def set_manual_seed(seed, device_type=None):
    if seed is None:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _seed_backend("mlu", "torch_mlu", seed, device_type)
    _seed_backend("npu", "torch_npu", seed, device_type)
    _seed_backend("musa", "torch_musa", seed, device_type)


def _seed_backend(attr_name, package_name, seed, device_type):
    if device_type not in (None, attr_name):
        return

    if device_type == attr_name:
        try:
            importlib.import_module(package_name)
        except ImportError:
            pass

    backend = getattr(torch, attr_name, None)
    if backend is None:
        return

    manual_seed_all = getattr(backend, "manual_seed_all", None)
    if callable(manual_seed_all):
        manual_seed_all(seed)
        return

    manual_seed = getattr(backend, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
