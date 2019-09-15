from typing import TypeVar, Callable
import time

A = TypeVar("A")


def measure(label: str, f: Callable[[], A], debug: bool = False) -> A:
    if debug:
        t = time.process_time()
        x = f()
        elapsed_time = time.process_time() - t
        print(label + " took " + str(elapsed_time))
        return x
    else:
        return f()
