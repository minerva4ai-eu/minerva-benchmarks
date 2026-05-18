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
            self.__dict__.setdefault(attr, []).append(time.time() - start)
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
            self.__dict__.setdefault(attr, []).append(time.perf_counter() - start)
            return result

        return wrapper

    return decorator
