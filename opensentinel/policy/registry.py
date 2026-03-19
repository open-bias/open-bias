"""
Policy engine registry for dynamic engine loading.

Provides a generic registry base class and a specialized registry
for policy engine implementations.
"""

from typing import Dict, Type, Optional, Any, List, Callable, TypeVar, Generic
import logging

from opensentinel.policy.protocols import PolicyEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GenericRegistry(Generic[T]):
    """
    Generic base class for type registries.

    Each subclass gets its own isolated ``_registry`` dict via
    ``__init_subclass__``, so registrations never leak between
    different registry types.
    """

    _registry: Dict[str, Type[T]]
    _label: str = "item"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, key: str, registered_class: Type[T]) -> None:
        if key in cls._registry:
            logger.warning(f"Overwriting existing {cls._label} registration: {key}")
        cls._registry[key] = registered_class
        logger.debug(f"Registered {cls._label}: {key}")

    @classmethod
    def get(cls, key: str) -> Optional[Type[T]]:
        return cls._registry.get(key)

    @classmethod
    def list_registered(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, key: str) -> bool:
        return key in cls._registry

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


class PolicyEngineRegistry(GenericRegistry[PolicyEngine]):
    """Registry for policy engine implementations."""

    _label = "policy engine"

    @classmethod
    def create(cls, engine_type: str) -> PolicyEngine:
        engine_class = cls.get(engine_type)
        if not engine_class:
            available = ", ".join(cls._registry.keys()) or "none"
            raise ValueError(
                f"Unknown policy engine type: '{engine_type}'. "
                f"Available engines: {available}"
            )
        return engine_class()

    @classmethod
    async def create_and_initialize(
        cls,
        engine_type: str,
        config: Dict[str, Any],
    ) -> PolicyEngine:
        engine = cls.create(engine_type)
        await engine.initialize(config)
        return engine

    # Backwards-compatible alias
    list_engines = GenericRegistry.list_registered


def register_engine(engine_type: str) -> Callable[[Type[PolicyEngine]], Type[PolicyEngine]]:
    """Decorator to auto-register a policy engine class."""

    def decorator(cls: Type[PolicyEngine]) -> Type[PolicyEngine]:
        PolicyEngineRegistry.register(engine_type, cls)
        return cls

    return decorator
