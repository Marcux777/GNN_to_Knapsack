"""
Registry system for models and decoders.

Provides global registries for discovering and creating models/decoders by name.
"""

from collections.abc import Callable
from typing import Any


class ModelRegistry:
    """
    Global registry for GNN models.

    Allows registration and creation of models by string name.
    Useful for configuration-driven model creation.

    Example:
        >>> @ModelRegistry.register("my_model")
        ... class MyModel(AbstractGNNModel):
        ...     pass
        ...
        >>> model = ModelRegistry.create("my_model", hidden_dim=64)
    """

    _models: dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a model class.

        Args:
            name: Unique name for the model

        Returns:
            Decorator function

        Example:
            >>> @ModelRegistry.register("pna")
            ... class KnapsackPNA(AbstractGNNModel):
            ...     pass
        """

        def wrapper(model_class: type) -> type:
            if name in cls._models:
                raise ValueError(f"Model '{name}' already registered as {cls._models[name]}")
            cls._models[name] = model_class
            return model_class

        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """
        Create model instance by name.

        Args:
            name: Registered model name
            **kwargs: Arguments to pass to model constructor

        Returns:
            Model instance

        Raises:
            KeyError: If model name not registered

        Example:
            >>> model = ModelRegistry.create("pna", hidden_dim=64, num_layers=3)
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """
        List all registered model names.

        Returns:
            List of model names

        Example:
            >>> print(ModelRegistry.list_models())
            ['pna', 'gcn', 'gat']
        """
        return list(cls._models.keys())

    @classmethod
    def get_class(cls, name: str) -> type:
        """
        Get model class by name without instantiating.

        Args:
            name: Model name

        Returns:
            Model class

        Example:
            >>> PNAClass = ModelRegistry.get_class("pna")
            >>> print(PNAClass.__doc__)
        """
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not registered")
        return cls._models[name]


class DecoderRegistry:
    """
    Global registry for solution decoders.

    Example:
        >>> @DecoderRegistry.register("greedy")
        ... class GreedyDecoder(AbstractDecoder):
        ...     pass
        ...
        >>> decoder = DecoderRegistry.create("greedy")
    """

    _decoders: dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a decoder class.

        Args:
            name: Unique name for the decoder

        Returns:
            Decorator function
        """

        def wrapper(decoder_class: type) -> type:
            if name in cls._decoders:
                raise ValueError(f"Decoder '{name}' already registered as {cls._decoders[name]}")
            cls._decoders[name] = decoder_class
            return decoder_class

        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """Create decoder instance by name."""
        if name not in cls._decoders:
            available = ", ".join(cls._decoders.keys())
            raise KeyError(f"Decoder '{name}' not found. Available: {available}")
        return cls._decoders[name](**kwargs)

    @classmethod
    def list_decoders(cls) -> list[str]:
        """List all registered decoder names."""
        return list(cls._decoders.keys())

    @classmethod
    def get_class(cls, name: str) -> type:
        """Get decoder class by name."""
        if name not in cls._decoders:
            raise KeyError(f"Decoder '{name}' not registered")
        return cls._decoders[name]
