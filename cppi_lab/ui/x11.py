"""What the X11 window manager can do (EWMH ``_NET_SUPPORTED``).

A window manager advertises the hints it honours in the ``_NET_SUPPORTED``
property of the root window.  The application asks before it relies on one:
a floating panel flagged as a utility window is simply never shown by a
window manager that does not know that window type (WSLg's is one).

The property is read through Xlib with ``ctypes`` on a short-lived
connection of its own, so nothing here depends on how Qt talks to the server.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from functools import lru_cache

UTILITY_WINDOW = "_NET_WM_WINDOW_TYPE_UTILITY"
_XA_ATOM = 4
_MAX_ATOMS = 4096


@lru_cache(maxsize=1)
def wm_supported() -> frozenset[str] | None:
    """Names of the EWMH hints the window manager of ``$DISPLAY`` advertises.

    An empty set means the server answered and the window manager advertises
    nothing; ``None`` means the question could not be asked (no display, no
    Xlib), in which case callers should assume a conventional window manager.
    """
    if not os.environ.get("DISPLAY"):
        return None
    try:
        xlib = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
    except OSError:
        return None
    pointer, atom, window = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong
    xlib.XOpenDisplay.restype = pointer
    xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    xlib.XCloseDisplay.argtypes = [pointer]
    xlib.XDefaultRootWindow.restype = window
    xlib.XDefaultRootWindow.argtypes = [pointer]
    xlib.XInternAtom.restype = atom
    xlib.XInternAtom.argtypes = [pointer, ctypes.c_char_p, ctypes.c_int]
    xlib.XGetAtomName.restype = pointer
    xlib.XGetAtomName.argtypes = [pointer, atom]
    xlib.XFree.argtypes = [pointer]
    xlib.XGetWindowProperty.restype = ctypes.c_int
    xlib.XGetWindowProperty.argtypes = [
        pointer,
        window,
        atom,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        atom,
        ctypes.POINTER(atom),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(pointer),
    ]
    display = xlib.XOpenDisplay(None)
    if not display:
        return None
    try:
        supported = xlib.XInternAtom(display, b"_NET_SUPPORTED", 1)
        if not supported:
            return frozenset()  # the atom does not exist: nobody ever set the property
        actual_type, actual_format = atom(), ctypes.c_int()
        count, remaining, data = ctypes.c_ulong(), ctypes.c_ulong(), pointer()
        status = xlib.XGetWindowProperty(
            display,
            xlib.XDefaultRootWindow(display),
            supported,
            0,
            _MAX_ATOMS,
            0,
            _XA_ATOM,
            ctypes.byref(actual_type),
            ctypes.byref(actual_format),
            ctypes.byref(count),
            ctypes.byref(remaining),
            ctypes.byref(data),
        )
        if status != 0 or not data or actual_format.value != 32:
            return frozenset()
        names = set()
        try:
            for value in ctypes.cast(data, ctypes.POINTER(atom))[: count.value]:
                name = xlib.XGetAtomName(display, value)
                if name:
                    names.add(ctypes.string_at(name).decode("ascii", "replace"))
                    xlib.XFree(name)
        finally:
            xlib.XFree(data)
        return frozenset(names)
    finally:
        xlib.XCloseDisplay(display)


def shows_utility_windows(platform_name: str) -> bool:
    """Whether a window flagged ``Qt.Tool`` will actually appear on this display.

    Only an X11 window manager that answers and does not list the utility
    window type is known not to show one; everything else is given the
    benefit of the doubt.
    """
    if platform_name != "xcb":
        return True
    supported = wm_supported()
    return supported is None or UTILITY_WINDOW in supported
