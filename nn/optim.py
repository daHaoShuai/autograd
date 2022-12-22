import numpy as np
from nn.nn import NeuralNet


class Optimizer:
    def zero_grad(self, net: NeuralNet) -> None:
        raise NotImplementedError

    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    随机梯度下降法(Stochastic Gradient Descent)
    """

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def zero_grad(self, net: NeuralNet) -> None:
        for _, grad in net.params_and_grads():
            grad = np.zeros_like(grad)

    def step(self, net: NeuralNet) -> None:
        # 更新权重
        for param, grad in net.params_and_grads():
            param -= self.lr * grad


class Momentum(Optimizer):
    """
    Momentum SGD
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def zero_grad(self, net: NeuralNet) -> None:
        for _, grad in net.params_and_grads():
            grad = np.zeros_like(grad)

    def step(self, net: NeuralNet) -> None:
        if self.v is None:
            self.v = {}
            for idx, (param, _) in enumerate(net.params_and_grads()):
                self.v[idx] = np.zeros_like(param)

        for idx, (param, grad) in enumerate(net.params_and_grads()):
            self.v[idx] = self.momentum * self.v[idx] - self.lr * grad
            param += self.v[idx]


class Adam(Optimizer):
    """
    Adam
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999) -> None:
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def zero_grad(self, net: NeuralNet) -> None:
        for _, grad in net.params_and_grads():
            grad = np.zeros_like(grad)

    def step(self, net: NeuralNet) -> None:
        if self.m is None and self.v is None:
            self.m, self.v = {}, {}
            for idx, (param, _) in enumerate(net.params_and_grads()):
                self.m[idx] = np.zeros_like(param)
                self.v[idx] = np.zeros_like(param)
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 **
                                 self.iter) / (1.0 - self.beta1**self.iter)

        for idx, (param, grad) in enumerate(net.params_and_grads()):
            self.m[idx] += (1 - self.beta1) * (grad - self.m[idx])
            self.v[idx] += (1 - self.beta2) * (grad ** 2 - self.v[idx])
            param -= lr_t * self.m[idx] / (np.sqrt(self.v[idx]) + 1e-7)
