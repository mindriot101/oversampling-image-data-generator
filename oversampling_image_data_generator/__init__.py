# Set up keras to use tensorflow
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
assert keras.__version__ == '2.1.5'
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
import itertools
import keras


def is_onehot(y):
    return len(y.shape) == 2

class OversamplingArrayIterator(NumpyArrayIterator):
    def __init__(self, *args, **kwargs):
        super(OversamplingArrayIterator, self).__init__(*args, **kwargs)

        if is_onehot(self.y):
            n_classes = self.y.shape[1]
            if n_classes != 2:
                raise ValueError('Only binary classification is supported')

            self.good_idx = np.where(self.y[:, 0] == 1)[0]
            self.defect_idx = np.where(self.y[:, 1] == 1)[0]
        else:
            n_classes = np.unique(self.y)
            if len(n_classes) != 2:
                raise ValueError('Only binary classification is supported')

            self.good_idx = np.where(self.y == 0)[0]
            self.defect_idx = np.where(self.y == 1)[0]

    def _flow_index(self):
        self.reset()
        self._set_index_array()
        while True:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)

            class_size = self.batch_size // 2

            defect_idx = np.random.choice(self.defect_idx, replace=False, size=class_size)
            good_idx = np.random.choice(self.good_idx, replace=False, size=class_size)
            idx = np.hstack([good_idx, defect_idx])
            idx = np.random.permutation(idx)

            self.total_batches_seen += 1
            yield idx


class OversamplingGenerator(ImageDataGenerator):
    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
            save_to_dir=None, save_prefix='', save_format='png', subset=None):
        return OversamplingArrayIterator(x, y, self,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                data_format=self.data_format,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format,
                subset=subset)
