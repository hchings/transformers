import psutil
import os

import torch
try:
    from py3nvml import py3nvml
except ImportError:
    py3nvml = None

dtype_to_bit = {
torch.float32 : 32,
torch.float64 : 64,
torch.float16: 16,
torch.bfloat16: 16,
torch.uint8: 8,
torch.int8: 8,
torch.int16: 16,
torch.int32: 32,
torch.int64: 64,
torch.bool: 1
}

process = psutil.Process(os.getpid())
base_mem_usage = process.memory_info().data
last_mem_usage = base_mem_usage

def memory_status(msg="", reset_max=True, sync=True):

    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])

    if sync:
        torch.cuda.synchronize()

    if global_rank != 0:
        return

    if py3nvml != None:
        py3nvml.nvmlInit()
        handle = py3nvml.nvmlDeviceGetHandleByIndex(local_rank)
        info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        total_used = info.used / 1024**3
        total_used_str = f"Totally used GPU memory: {total_used}"
    else:
        total_used_str = ""

    alloced = torch.cuda.memory_allocated(device=local_rank)
    max_alloced = torch.cuda.max_memory_allocated(device=local_rank)
    cached = torch.cuda.memory_reserved(device=local_rank)
    max_cached = torch.cuda.max_memory_reserved(device=local_rank)

    # convert to GB for printing
    alloced /= 1024**3
    cached /= 1024**3
    max_alloced /= 1024**3
    max_cached /= 1024**3

    print(
        f'[{msg}] rank {global_rank}',
        f'device={local_rank} '
        f'alloc {alloced:0.4f} max_alloced {max_alloced:0.4f} '
        f'cache {cached:0.4f} max_cached {max_cached:0.4f} '
        f'{total_used_str}'
    )
    if reset_max:
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()
    if py3nvml != None:
        py3nvml.nvmlShutdown()
