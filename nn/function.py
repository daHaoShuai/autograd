import numpy as np
from nn.tensor import Tensor


def tanh(x: Tensor) -> Tensor:
    """
    tanh函数
    """
    return np.tanh(x)


def tanh_grad(x: Tensor) -> Tensor:
    """
    tanh函数求导
    """
    y = tanh(x)
    return 1 - y ** 2


def sigmoid(x: Tensor) -> Tensor:
    """
    sigmoid函数
    """
    return 1 / (1 + np.exp(-x))


def relu(x: Tensor) -> Tensor:
    """
    relu函数
    """
    return np.maximum(0, x)


def relu_grad(x: Tensor) -> Tensor:
    """
    relu函数求导
    """
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad


def softmax(x: Tensor) -> Tensor:
    """
    softmax函数
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y: Tensor, t: Tensor) -> Tensor:
    """
    均方损失函数
    """
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y: Tensor, t: Tensor) -> Tensor:
    """
    交叉熵损失函数
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
