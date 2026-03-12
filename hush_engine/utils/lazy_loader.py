"""
Lazy-loading utilities for Hush Engine v1.8.0.

Provides decorators and descriptors for deferring heavy module imports and
object initialization until first access. This reduces idle RAM and startup time
by avoiding eager loading of detectors, NER models, and data files that may not
be needed for every detection request.
"""

import threading
from functools import wraps


class lazy_property:
    """Descriptor that converts a method into a lazy-evaluated cached property.

    Thread-safe: uses a lock to prevent concurrent initialization.

    Usage:
        class MyClass:
            @lazy_property
            def heavy_detector(self):
                from heavy_module import HeavyDetector
                return HeavyDetector()
    """

    def __init__(self, func):
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"
        self.lock = threading.Lock()
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self.attr_name = f"_lazy_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # Fast path: already initialized
        val = getattr(obj, self.attr_name, None)
        if val is not None:
            return val
        # Slow path: initialize with lock
        with self.lock:
            val = getattr(obj, self.attr_name, None)
            if val is not None:
                return val  # Another thread initialized it
            val = self.func(obj)
            setattr(obj, self.attr_name, val)
            return val


def lazy_import(module_path: str, attr: str = None):
    """Create a lazy module/attribute import that loads on first access.

    Args:
        module_path: Dotted module path (e.g., "hush_engine.detectors.face_detector")
        attr: Optional attribute name to import from the module

    Returns:
        A proxy object that imports on first attribute access.

    Usage:
        FaceDetector = lazy_import("hush_engine.detectors.face_detector", "FaceDetector")
        # Module is not imported yet
        detector = FaceDetector()  # Now it imports
    """
    _module = None
    _lock = threading.Lock()

    class LazyProxy:
        def __getattr__(self, name):
            nonlocal _module
            if _module is None:
                with _lock:
                    if _module is None:
                        import importlib
                        mod = importlib.import_module(module_path)
                        _module = getattr(mod, attr) if attr else mod
            return getattr(_module, name)

        def __call__(self, *args, **kwargs):
            nonlocal _module
            if _module is None:
                with _lock:
                    if _module is None:
                        import importlib
                        mod = importlib.import_module(module_path)
                        _module = getattr(mod, attr) if attr else mod
            return _module(*args, **kwargs)

        def __repr__(self):
            target = f"{module_path}.{attr}" if attr else module_path
            if _module is not None:
                return repr(_module)
            return f"<lazy: {target}>"

    return LazyProxy()
