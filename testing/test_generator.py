import numpy as np
from gentest import OversamplingGenerator


def test_onehot():
    n_images = 200
    shape = (200, 32, 32, 1)
    X = np.random.uniform(0., 1., size=shape)
    y = np.zeros((n_images, 2), dtype=int)
    y[:20, 1] = 1
    y[20:, 0] = 1

    batch_size = 32
    generator = OversamplingGenerator()
    gen = generator.flow(X, y, batch_size=batch_size)

    n_iter = 10

    for _ in range(n_iter):
        _, y_batch = next(gen)
        idx = np.where(y_batch[:, 0] == 0)[0]
        n_0 = idx.size
        n_target = batch_size // 2

        msg = f'Class size for 0 is {n_0}, should be {n_target}'

        assert  n_0 == batch_size // 2, msg

def test_not_onehot():
    n_images = 200
    shape = (200, 32, 32, 1)
    X = np.random.uniform(0., 1., size=shape)
    y = np.zeros(n_images, dtype=int)
    y[:20] = 1

    batch_size = 32
    generator = OversamplingGenerator()
    gen = generator.flow(X, y, batch_size=batch_size)

    n_iter = 10

    for _ in range(n_iter):
        _, y_batch = next(gen)
        idx = np.where(y_batch == 0)[0]
        n_0 = idx.size
        n_target = batch_size // 2

        msg = f'Class size for 0 is {n_0}, should be {n_target}'

        assert  n_0 == batch_size // 2, msg
