import numpy as np
from .State import State


def test():
    A = np.arange(16).reshape(4, 2, 2)
    xyt = np.arange(3 * 16).reshape(3, 4, 2, 2)
    myu = np.arange(16).reshape(4, 2, 2)

    state = State(state=A, xyt=xyt, myu=myu, from_simulation=True)
    a = A.reshape(-1)
    # Adjust this based on expected behavior of the generate_batches method
    expected_batches = [
        a[0:3], a[3:6], a[6:9], a[9:12], a[12:15], 
        np.concatenate([a[15:], a[:2]])
    ]

    for i, s in enumerate(state.generate_batches(nbatches=6, batch_size=3, verbose=0, shuffle=False)):
        assert np.array_equal(s.state, expected_batches[i]), f"Batch {i} does not match expected values."
    


    return "Test succeeded"
