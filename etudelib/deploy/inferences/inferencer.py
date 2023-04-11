

class Inferencer(ABC):

    @abstractmethod
    def load_model(self, path: str | Path) -> Any:
        """Load Model."""
        raise  NotImplementedError