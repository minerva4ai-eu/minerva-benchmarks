import functools
import logging
import time

logger = logging.getLogger(__name__)


def timed(attr: str):
    """
    Method decorator that appends the execution time (seconds) of the
    decorated method to the list `self.<attr>` after each call.

    Usage:
        @timed("step_times")
        def training_step(self, ...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            duration = time.time() - start
            self.__dict__.setdefault(attr, []).append(duration)
            print(f"[TIMING] {func.__name__} took {duration:.4f} seconds to run.")
            return result

        return wrapper

    return decorator


def perf_timed(attr: str):
    """
    Method decorator that appends the execution time (seconds) of the
    decorated method to the list `self.<attr>` after each call.

    Usage:
        @perf_timed("step_times")
        def training_step(self, ...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            duration = time.perf_counter() - start
            self.__dict__.setdefault(attr, []).append(duration)
            print(f"[TIMING] {func.__name__} took {duration:.4f} seconds to run.")
            return result

        return wrapper

    return decorator
