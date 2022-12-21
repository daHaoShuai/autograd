from typing import Iterator, NamedTuple
import numpy as np
from nn.tensor import Tensor

Batch = NamedTuple('Batch', [('inputs', Tensor), ('targets', Tensor)])


class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    """
    batch_size : 一批数据的大小
    shuffle : 是否打乱数据
    """
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        # 挑出batch_size个数
        starts = np.arange(0, len(inputs), self.batch_size)
        # 看看要不要随机打乱
        if self.shuffle:
            np.random.shuffle(starts)
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
