"""
Policy compiler registry for dynamic compiler loading.

Provides a central registry for policy compiler implementations,
enabling dynamic selection and instantiation of compilers.

Mirrors PolicyEngineRegistry pattern.
"""

from typing import Dict, Type, Optional, List, Callable
import logging

from opensentinel.policy.compiler.protocol import PolicyCompiler

logger = logging.getLogger(__name__)


class PolicyCompilerRegistry:
    """
    Registry for policy compiler implementations.

    Allows dynamic registration and lookup of policy compilers.
    Compilers are registered by engine type (e.g., 'fsm', 'nemo').

    Usage:
        # Registration (usually via decorator)
        PolicyCompilerRegistry.register("fsm", FSMCompiler)

        # Lookup
        compiler_class = PolicyCompilerRegistry.get("fsm")
    """

    _compilers: Dict[str, Type[PolicyCompiler]] = {}

    @classmethod
    def register(cls, engine_type: str, compiler_class: Type[PolicyCompiler]) -> None:
        """
        Register a policy compiler class.

        Args:
            engine_type: Engine type identifier (e.g., 'fsm', 'nemo')
            compiler_class: The compiler class to register
        """
        if engine_type in cls._compilers:
            logger.warning(
                f"Overwriting existing policy compiler registration: {engine_type}"
            )
        cls._compilers[engine_type] = compiler_class
        logger.debug(f"Registered policy compiler: {engine_type}")

    @classmethod
    def get(cls, engine_type: str) -> Optional[Type[PolicyCompiler]]:
        """
        Get a policy compiler class by engine type.

        Args:
            engine_type: Engine type identifier

        Returns:
            Compiler class or None if not found
        """
        return cls._compilers.get(engine_type)

    @classmethod
    def list_compilers(cls) -> List[str]:
        """
        List all registered compiler engine types.

        Returns:
            List of engine type identifiers with registered compilers
        """
        return list(cls._compilers.keys())

    @classmethod
    def is_registered(cls, engine_type: str) -> bool:
        """
        Check if a compiler is registered for an engine type.

        Args:
            engine_type: Engine type identifier

        Returns:
            True if registered, False otherwise
        """
        return engine_type in cls._compilers

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registrations.

        Mainly useful for testing.
        """
        cls._compilers.clear()


def register_compiler(engine_type: str) -> Callable[[Type[PolicyCompiler]], Type[PolicyCompiler]]:
    """
    Decorator to auto-register a policy compiler class.

    Usage:
        @register_compiler("fsm")
        class FSMCompiler(PolicyCompiler):
            ...

    Args:
        engine_type: Engine type identifier for the compiler

    Returns:
        Decorator function
    """

    def decorator(cls: Type[PolicyCompiler]) -> Type[PolicyCompiler]:
        PolicyCompilerRegistry.register(engine_type, cls)
        return cls

    return decorator
