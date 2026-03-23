"""
Policy compiler registry for dynamic compiler loading.

Thin subclass of GenericRegistry specialized for PolicyCompiler.
"""

from typing import Type, Callable

from openbias.policy.registry import GenericRegistry
from openbias.policy.compiler.protocol import PolicyCompiler


class PolicyCompilerRegistry(GenericRegistry[PolicyCompiler]):
    """Registry for policy compiler implementations."""

    _label = "policy compiler"

    @classmethod
    def list_compilers(cls) -> list[str]:
        return cls.list_registered()


def register_compiler(engine_type: str) -> Callable[[Type[PolicyCompiler]], Type[PolicyCompiler]]:
    """Decorator to auto-register a policy compiler class."""

    def decorator(cls: Type[PolicyCompiler]) -> Type[PolicyCompiler]:
        PolicyCompilerRegistry.register(engine_type, cls)
        return cls

    return decorator
