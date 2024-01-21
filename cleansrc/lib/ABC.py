from abc import ABC, abstractmethod
import torch
import numpy as np

class ABCState(ABC):
    def __init__(self, *, from_simulation=False):
        super().__init__()

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def get_state(self):
        # Returns a 4D Tensor
        pass
    def generate_batches(nbatches, batch_size):
        pass

    @abstractmethod
    def get_imag(self):
        pass

    @abstractmethod
    def get_real(self):
        pass

    @abstractmethod
    def get_pair(self):
        pass

    @abstractmethod
    def get_angle(self):
        pass

    @abstractmethod
    def get_abs(self):
        pass

    @abstractmethod
    def get_abs_squared(self):
        pass

    @abstractmethod
    def get_ri_multiplication(self):
        pass

    @abstractmethod
    def get_2d_tensor_state(self) -> torch.tensor:
        pass

    @abstractmethod
    def get_2d_tensor_xyt(self) -> torch.tensor:
        pass

    @abstractmethod
    def set_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def update_state(self):
        pass

    @property
    @abstractmethod
    def state(self):
        pass

    @state.setter
    @abstractmethod
    def state(self, value):
        pass

    @property
    @abstractmethod
    def real(self):
        pass

    @property
    @abstractmethod
    def imag(self, value):
        pass

    @property
    @abstractmethod
    def vector(self):
        pass

    @property
    @abstractmethod
    def state(self):
        pass

    @property
    @abstractmethod
    def pair(self):
        pass

    @property
    @abstractmethod
    def angle(self):
        pass

    @property
    @abstractmethod
    def abs(self):
        pass

    @property
    @abstractmethod
    def abs_squared(self):
        pass

    @property
    @abstractmethod
    def ri(self):
        pass

    @property
    @abstractmethod
    def ri_multiplication(self):
        pass

    @property
    @abstractmethod
    def _2d_tensor(self):
        pass
