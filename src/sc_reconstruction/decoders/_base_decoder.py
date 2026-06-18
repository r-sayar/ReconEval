from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class BaseReconstructionDecoder(ABC):
    """
    Abstract base class for single-cell reconstruction decoders.
    Subclasses must implement `decode`, `save` and `load`.
    """

    @abstractmethod
    def decode(self, latent: Any, **kwargs) -> Any:
        """
        Decode the latent representation back.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained decoder.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained decoder.
        """
        pass