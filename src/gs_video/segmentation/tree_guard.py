from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import threading
from ctypes import wintypes
from typing import Any, Protocol


class ProcessTreeGuard(Protocol):
    def terminate(self, *, force: bool) -> bool: ...

    def close(self) -> None: ...


class NoopTreeGuard:
    def terminate(self, *, force: bool) -> bool:
        del force
        return False

    def close(self) -> None:
        return


class PosixProcessGroupGuard:
    def __init__(self, pid: int) -> None:
        self._pgid = pid
        self._closed = False

    def terminate(self, *, force: bool) -> bool:
        if self._closed:
            return False
        killpg = getattr(os, "killpg", None)
        if not callable(killpg):
            return False
        sig = getattr(signal, "SIGKILL" if force else "SIGTERM", 9 if force else 15)
        try:
            killpg(self._pgid, sig)
        except ProcessLookupError:
            return False
        return True

    def close(self) -> None:
        if self._closed:
            return
        self.terminate(force=True)
        self._closed = True


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsJobObjectGuard:
    _KILL_ON_JOB_CLOSE = 0x00002000
    _EXTENDED_LIMIT_INFORMATION_CLASS = 9

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._lock = threading.Lock()
        self._pid = process.pid
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle: int | None = int(handle)
        try:
            information = _ExtendedLimitInformation()
            information.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
            if not self._kernel32.SetInformationJobObject(
                handle,
                self._EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            process_handle = getattr(process, "_handle", None)
            if process_handle is None or not self._kernel32.AssignProcessToJobObject(
                wintypes.HANDLE(handle), wintypes.HANDLE(int(process_handle))
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self._kernel32.CloseHandle(handle)
            self._handle = None
            raise

    def terminate(self, *, force: bool) -> bool:
        with self._lock:
            if self._handle is None:
                return False
            if force:
                return bool(self._kernel32.TerminateJobObject(wintypes.HANDLE(self._handle), 1))
            try:
                completed = subprocess.run(
                    ["taskkill", "/PID", str(self._pid), "/T"],
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            return completed.returncode == 0

    def close(self) -> None:
        with self._lock:
            if self._handle is None:
                return
            handle = self._handle
            self._handle = None
            if not self._kernel32.CloseHandle(wintypes.HANDLE(handle)):
                raise ctypes.WinError(ctypes.get_last_error())


def create_process_tree_guard(process: Any) -> ProcessTreeGuard:
    if not isinstance(process, subprocess.Popen):
        return NoopTreeGuard()
    if os.name == "nt":
        return WindowsJobObjectGuard(process)
    return PosixProcessGroupGuard(process.pid)
