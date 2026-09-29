"""Opt this process out of Windows power throttling (EcoQoS / efficiency mode).

Servers launched hidden via WMI are classed as background work, and Windows then runs their CPU threads
at efficiency clocks: the ORT/torch CPU paths measured ~4x slower than the same code in a foreground
process. Importing this module first restores full speed. No-op off Windows.
"""
import ctypes, sys

def disable():
    if sys.platform != "win32":
        return False
    from ctypes import wintypes

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    EXECUTION_SPEED, IGNORE_TIMER_RESOLUTION, ProcessPowerThrottling = 0x1, 0x4, 4
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    # control both flags, state 0 = never throttle
    st = PROCESS_POWER_THROTTLING_STATE(1, EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, 0)
    ok = k32.SetProcessInformation(k32.GetCurrentProcess(), ProcessPowerThrottling, ctypes.byref(st), ctypes.sizeof(st))
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000)   # ABOVE_NORMAL_PRIORITY_CLASS
    print(f"[nothrottle] power throttling disabled: {bool(ok)}", flush=True)
    return bool(ok)

disable()
