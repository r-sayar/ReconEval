# models/base/_base_model.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseReconstructionModel(ABC):
    """
    Abstract base class for single-cell reconstruction models.
    Subclasses must implement `prepare`, `train`, `predict`, `save` and `load`.
    """

    @abstractmethod
    def prepare(self, data: Any) -> None:
        """
        Perform any setup steps needed before training, 
        such as data preprocessing or model initialization.
        """
        pass

    @abstractmethod
    def train(self, max_epochs: int = 2, **kwargs) -> None:
        """
        Train the model. 
        `max_epochs` or other training parameters can be passed here.
        """
        pass

    @abstractmethod
    def predict(self, combos: list[tuple], **kwargs) -> Dict[str, Any]:
        """
        Run inference in identifier level (e.g., cell line X drug X dose),
        and return a dictionary mapping each identifier
        to its predicted result.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained model.
        """
        pass