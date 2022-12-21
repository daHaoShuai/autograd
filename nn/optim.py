from nn.nn import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    梯度下降
    """
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        # 更新权重
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
