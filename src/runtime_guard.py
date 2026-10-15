"""Single-GPU memory budget for HW1 commands; CPU/MPS need no VRAM guard."""
import os
import threading
import time


def start_guard(device):
    if not str(device).isdigit(): return
    if str(device) != "0": raise ValueError("Use one visible CUDA GPU as device0")
    import torch
    torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,10.5*1024**3/total),0)
    import pynvml
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    def watch():
        failures=0;last_ok=time.monotonic()
        while True:
            try:
                used=pynvml.nvmlDeviceGetMemoryInfo(handle).used/1024**2
                failures=0;last_ok=time.monotonic()
                if used>11400:
                    print('HW1 memory guard: VRAM exceeds11400MiB',flush=True);os._exit(91)
            except Exception as e:
                failures+=1
                if failures>=3 or time.monotonic()-last_ok>3:
                    print(f"HW1 memory guard: {e}",flush=True);os._exit(91)
            threading.Event().wait(.25)
    threading.Thread(target=watch,daemon=True).start()
