from autograd.module import Module


class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, model: Module) -> None:
        """
        更新权重
        """
        for parameter in model.parameters():
            parameter -= parameter.grad * self.lr
