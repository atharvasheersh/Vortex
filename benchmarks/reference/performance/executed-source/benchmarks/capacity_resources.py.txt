"""Optional resident working-set bound for this benchmark process on Windows."""

import sys
from contextlib import contextmanager


@contextmanager
def working_set_limit(mebibytes):
    if not mebibytes:
        yield {"enabled": False, "scope": "OS-managed process working set"}
        return
    if sys.platform != "win32":
        raise OSError("--working-set-mib requires Windows; no bound was applied")
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessWorkingSetSizeEx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    kernel.SetProcessWorkingSetSizeEx.restype = ctypes.c_int
    kernel.GetProcessWorkingSetSizeEx.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    kernel.GetProcessWorkingSetSizeEx.restype = ctypes.c_int
    process = kernel.GetCurrentProcess()

    def read():
        minimum, maximum, flags = (
            ctypes.c_size_t(),
            ctypes.c_size_t(),
            ctypes.c_uint32(),
        )
        if not kernel.GetProcessWorkingSetSizeEx(
            process, ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(flags)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return minimum.value, maximum.value, flags.value

    previous = read()
    maximum = mebibytes * 1024**2
    # HARDWS_MAX_ENABLE | HARDWS_MIN_DISABLE; never reserve a hard minimum.
    if not kernel.SetProcessWorkingSetSizeEx(process, 1024**2, maximum, 0x4 | 0x2):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        actual_min, actual_max, flags = read()
        if actual_max != maximum or not flags & 0x4 or flags & 0x8:
            raise RuntimeError(
                "Windows did not confirm the requested working-set bound"
            )
        yield {
            "enabled": True,
            "scope": "this process resident working set only; not commit or total OS file cache",
            "minimum_bytes": actual_min,
            "maximum_bytes": actual_max,
            "flags": flags,
            "method": "SetProcessWorkingSetSizeEx HARDWS_MAX_ENABLE; verified by GetProcessWorkingSetSizeEx",
        }
    finally:
        restore_flags = (0x1 if previous[2] & 0x1 else 0x2) | (
            0x4 if previous[2] & 0x4 else 0x8
        )
        if not kernel.SetProcessWorkingSetSizeEx(
            process, previous[0], previous[1], restore_flags
        ):
            raise ctypes.WinError(ctypes.get_last_error())
