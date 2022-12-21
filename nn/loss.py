import numpy as np
from nn.tensor import Tensor
from nn.function import softmax, mean_squared_error, cross_entropy_error


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    均方损失
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return mean_squared_error(predicted, actual)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)


class CrossEntropyLoss(Loss):
    """
    交叉熵损失
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return cross_entropy_error(softmax(predicted), actual)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        batch_size = predicted.shape[0]
        # 监督数据是one-hot-vector的情况
        if actual.size == predicted.size:
            dx = (predicted - actual) / batch_size
        else:
            dx = predicted.copy()
            dx[np.arange(batch_size), actual] -= 1
            dx = dx / batch_size
        return dx
