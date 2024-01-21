import numpy as np
from tqdm import tqdm
import math
import torch

class State:
    def __init__(self, state, xyt=None, myu=None, slice=None, shuffled_indices=None):
        self.state = state
        self.xyt = xyt
        self.myu = myu
        self.slice = slice
        self.shuffled_indices = shuffled_indices

    def __len__(self):
        return len(self.state)
    
    @property
    def shape(self):
        return self.state.shape

    def __getitem__(self, slice):
        return State(
            state = self.state[slice],
            xyt = self.xyt[:,slice],
            myu = self.myu[slice] if self.myu is not None else None ,
            slice = slice,
            shuffled_indices = self.shuffled_indices
        )
    
    def flatten(self):
        return State(
            state = self.state.flatten(),
            xyt = self.xyt.flatten().reshape(3,-1),
            myu = self.myu.flatten() if self.myu is not None else None,
            slice = self.slice,
            shuffled_indices = self.shuffled_indices
        )
    
    def append(self, state):
        if not isinstance(state, State):
            raise TypeError("The argument must be an instance of State")

        self.state = np.append(self.state, state.state, axis=0)
        self.xyt = np.append(self.xyt, state.xyt, axis=1)
        self.myu = torch.cat((self.myu, state.myu), dim=0)
        return self

    def shuffle(self):
        # Generate shuffled indices
        shuffled_indices = np.arange(len(self.state))
        np.random.shuffle(shuffled_indices)

        # Shuffle each attribute using the same shuffled indices
        shuffled_state = self.state[shuffled_indices]
        shuffled_xyt = self.xyt[:, shuffled_indices]
        shuffled_myu = self.myu[torch.tensor(shuffled_indices)] if self.myu is not None else None 

        return State(shuffled_state, shuffled_xyt, shuffled_myu, slice = self.slice, shuffled_indices = shuffled_indices)

    def generate_batches(self, nbatches, batch_size, verbose=0, shuffle=True):
        flatself = self.flatten()
        buffer = flatself.shuffle() if shuffle else flatself

        iterable = range(nbatches)
        if verbose != 0:
            iterable = tqdm(iterable)

        index = 0

        for _ in iterable:
            while len(buffer) < index + batch_size:
                buffer.append(flatself.shuffle() if shuffle else flatself)
                if verbose == 2:
                    print("Next full iteration over the entire dataset")
            yield buffer[index:index+batch_size]
            index = index + batch_size
            if verbose == 2 and index >= len(buffer):
                print("Next full iteration over the entire dataset")
            index = index % len(buffer)


    def iterate_over_data(self, batch_size, verbose=0, shuffle=True):
        flatself = self.flatten()
        buffer = flatself.shuffle() if shuffle else flatself        

        iterator = range(0, len(buffer), batch_size)
        if verbose == 1:
            iterator = tqdm(iterator, total=len(iterator), desc="Processing")

        for start_idx in iterator:
            yield buffer[start_idx:start_idx + batch_size]
            if verbose == 1:
                iterator.set_description(f"Processed up to index {start_idx + batch_size}")

        
    def get_2d_tensor_state(self, iscomplex = False, **kwargs):
        if iscomplex:
            return torch.tensor(self.state, dtype=torch.complex128, **kwargs)
        else:
            return torch.tensor(np.hstack([self.state.real[:, None],self.state.imag[:, None]]), dtype=torch.float64, **kwargs).T

    def get_2d_tensor_xyt(self, **kwargs):
        return torch.tensor(self.xyt, dtype=torch.float64, **kwargs)

    def get_2d_tensor_state_lp(self, iscomplex = False, **kwargs):
        if iscomplex:
            return torch.tensor(self.state, dtype=torch.complex64, **kwargs)
        else:
            return torch.tensor(np.hstack([self.state.real[:, None],self.state.imag[:, None]]), dtype=torch.float32, **kwargs).T

    def get_2d_tensor_xyt_lp(self, **kwargs):
        return torch.tensor(self.xyt, dtype=torch.float32, **kwargs)
    
    def get_2d_tensor_state_llp(self, iscomplex = False, **kwargs):
        if iscomplex:
            return torch.tensor(self.state, dtype=torch.complex32, **kwargs)
        else:
            return torch.tensor(np.hstack([self.state.real[:, None],self.state.imag[:, None]]), dtype=torch.float16, **kwargs).T

    def get_2d_tensor_xyt_llp(self, **kwargs):
        return torch.tensor(self.xyt, dtype=torch.float16, **kwargs)