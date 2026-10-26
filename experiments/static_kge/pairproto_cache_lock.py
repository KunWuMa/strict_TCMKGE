"""Waitable per-artifact locks for independent, resumable local computations."""
from contextlib import contextmanager
from functools import wraps
import inspect
from pathlib import Path
import time
from pairproto_execution_lock import ExclusiveLock

OUT=Path(__file__).resolve().parent/'results'


@contextmanager
def cache_lock(tag):
    while True:
        try:
            lock=ExclusiveLock(OUT/(tag+'.lock'));lock.__enter__();break
        except RuntimeError:time.sleep(2)
    try:yield
    finally:lock.__exit__(None,None,None)


def serialized_cache(role,tag_factory):
    def decorate(function):
        signature=inspect.signature(function)
        @wraps(function)
        def call(*args,**kwargs):
            values=signature.bind(*args,**kwargs);values.apply_defaults()
            with cache_lock(tag_factory(values.arguments)+'.'+role):
                return function(*args,**kwargs)
        return call
    return decorate
