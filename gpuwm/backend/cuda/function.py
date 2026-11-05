"""Selected GPU backend compatibility namespace.

A name this module cannot serve (no local GPU, no backend) answers
``AttributeError`` as PEP 562 requires, so ``getattr(module, name, default)``
and module walks work; see ``gpuwm.backend.cuda.__init__``.
"""
from gpuwm.backend import array_module


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        function = array_module().cuda.function
    except Exception as error:
        raise AttributeError(f"gpuwm.backend.cuda.function.{name}: {error}") from error
    return getattr(function, name)
