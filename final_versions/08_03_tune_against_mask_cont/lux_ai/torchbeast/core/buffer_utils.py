from copy import copy
import gym
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Tuple, Union
import os

Buffers = List[Dict[str, Union[Dict, torch.Tensor]]]


def fill_buffers_inplace(buffers, fill_vals, non_blocking=False):
    if isinstance(fill_vals, dict):
        if set(buffers.keys()) != set(fill_vals.keys()):
            print(f'! fill_buffers_inplace keys differ: {set(buffers.keys())} != {set(fill_vals.keys())}', flush=True)
            assert False
        for key, val in fill_vals.items():
            assert key in buffers
            fill_buffers_inplace(buffers[key], val, non_blocking=non_blocking)
    else:
        #print("SHAPES:", buffers.shape, fill_vals.shape)
        buffers.copy_(fill_vals, non_blocking=non_blocking)


def fill_buffers_inplace_2(buffers: Union[Dict, torch.Tensor], fill_vals: Union[Dict, torch.Tensor], step: int):
    if isinstance(fill_vals, dict):
        for key, val in fill_vals.items():
            if key not in buffers:
                continue
            fill_buffers_inplace_2(buffers[key], val, step)
    else:
        buffers[step, ...] = fill_vals[:]


def fill_buffers_inplace_3(buffers: Union[Dict, torch.Tensor], fill_vals: Union[Dict, torch.Tensor], a, b):
    if isinstance(fill_vals, dict):
        for key, val in fill_vals.items():
            if key not in buffers:
                continue
            fill_buffers_inplace_3(buffers[key], val, a, b)
    else:
        buffers[:,a:b, ...] = fill_vals[:]


def stack_buffers(buffers: Buffers, dim: int) -> Dict[str, Union[Dict, torch.Tensor]]:
    stacked_buffers = {}
    for key, val in copy(buffers[0]).items():
        if isinstance(val, dict):
            stacked_buffers[key] = stack_buffers([b[key] for b in buffers], dim)
        else:
            stacked_buffers[key] = torch.cat([b[key] for b in buffers], dim=dim)
    return stacked_buffers


def split_buffers(
        buffers: Dict[str, Union[Dict, torch.Tensor]],
        split_size_or_sections: Union[int, List[int]],
        dim: int,
        contiguous: bool,
) -> List[Union[Dict, torch.Tensor]]:
    buffers_split = None
    for key, val in copy(buffers).items():
        if isinstance(val, dict):
            bufs = split_buffers(val, split_size_or_sections, dim, contiguous)
        else:
            bufs = torch.split(val, split_size_or_sections, dim=dim)
            if contiguous:
                bufs = [b.contiguous() for b in bufs]

        if buffers_split is None:
            buffers_split = [{} for _ in range(len(bufs))]
        assert len(bufs) == len(buffers_split)
        buffers_split = [dict(**{key: buf}, **d) for buf, d in zip(bufs, buffers_split)]
    return buffers_split


def buffers_apply(buffers: Union[Dict, torch.Tensor], func: Callable[[torch.Tensor], Any]) -> Union[Dict, torch.Tensor]:
    if isinstance(buffers, dict):
        return {
            key: buffers_apply(val, func) for key, val in copy(buffers).items()
        }
    else:
        return func(buffers)


def checksum_buffers(buffers: Union[Dict, torch.Tensor]):
    if isinstance(buffers, dict):
        sum = 0.
        for key, val in buffers.items():
            if key != 'checksum_GPU_CPU' and key != 'storage_id_GPU_CPU':
                sum += checksum_buffers(val)
            #print(f"checksum: {key}, {checksum_buffers(val)}")
        return sum
    else:
        sum = buffers.sum()
        if torch.isneginf(sum):
            sum = 0.
        return sum



def get_gpu_buffers(buffers: Union[Dict, torch.Tensor], transfer=False, device=None):
    if isinstance(buffers, dict):
        result = {
            key: get_gpu_buffers(val, transfer or "GPU" in key, device) for key, val in copy(buffers).items()
        }
        keys = list(result.keys())
        for key in keys:
            if result[key] is None:
                result.pop(key)
        if len(result) == 0:
            return None
        return result
    else:
        if transfer:
            if os.getenv("MAC") == "1":
                #if buffers.dtype == torch.float32:  # Fix for PyTorch tensors
                #    return buffers.to(dtype=torch.bfloat16, device='cpu', non_blocking=False)
                return buffers.to('cpu', non_blocking=False)
            else:
                #if buffers.dtype == torch.float32:  # Fix for PyTorch tensors
                #    return buffers.to(dtype=torch.bfloat16, device='cpu', non_blocking=False)
                return buffers.to(device, non_blocking=False)

        return None

def get_cpu_buffers(buffers: Union[Dict, torch.Tensor], transfer=False):
    if isinstance(buffers, dict):
        result = {
            key: get_cpu_buffers(val, transfer or "CPU" in key) for key, val in copy(buffers).items()
        }
        keys = list(result.keys())
        for key in keys:
            if result[key] is None:
                result.pop(key)
        if len(result) == 0:
            return None
        return result
    else:
        if transfer:
            #if buffers.dtype == torch.float32:  # Fix for PyTorch tensors
            #    return buffers.to(dtype=torch.bfloat16, device='cpu', non_blocking=False)
            return buffers.to('cpu', non_blocking=False)
        return None


def buffers_del(buffers: Union[Dict, torch.Tensor], device=None) -> None:
    if isinstance(buffers, dict):
         for key, val in copy(buffers).items():
             buffers_del(val, device)
    else:
        if buffers.device == device:
            del buffers


def _create_buffers_from_specs(specs: Dict[str, Union[Dict, Tuple, torch.dtype]]) -> Union[Dict, torch.Tensor]:
    if isinstance(specs, dict) and "dtype" not in specs.keys():
        new_buffers = {}
        for key, val in specs.items():
            new_buffers[key] = _create_buffers_from_specs(val)
        return new_buffers
    else:
        if os.getenv("MAC") == "1":
            return torch.empty(**specs).to('cpu', non_blocking=True)
        else:
            return torch.empty(**specs).to('cuda', non_blocking=True)


def _create_buffers_like(buffers: Union[Dict, torch.Tensor], t_dim: int, is_cpu: True) -> Union[Dict, torch.Tensor]:
    if isinstance(buffers, dict):
        torch_buffers = {}
        for key, val in buffers.items():
            torch_buffers[key] = _create_buffers_like(val, t_dim, key.startswith("LOGGING"))
        return torch_buffers
    else:
        buffers = buffers.unsqueeze(0).expand(t_dim, *[-1 for _ in range(len(buffers.shape))])
        if is_cpu:
            return torch.empty_like(buffers).share_memory_()
        else:
            if os.getenv("MAC") == "1":
                return torch.empty_like(buffers).to('cpu', non_blocking=True)
            else:
                return torch.empty_like(buffers).to('cuda', non_blocking=True)


def buffers_print(buffers: Union[Dict, torch.Tensor], total_path: str = ""):
    if isinstance(buffers, dict):
        for key, val in buffers.items():
            buffers_print(val, total_path + f"/{key}")
    else:
        print(f"buffer: {total_path} {buffers.shape} {buffers.dtype} {buffers.device}")


def check_device_cpu(buffers: Union[Dict, torch.Tensor], key=None):
    if isinstance(buffers, dict):
        for key, val in buffers.items():
            check_device_cpu(val, key)
    else:
        assert buffers.device == torch.device('cpu'), f"{key}: {buffers.device}"

