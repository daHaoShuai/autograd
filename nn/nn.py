from typing import Sequence, Iterable, Tuple
from nn.tensor import Tensor
from nn.layers import Layer


class NeuralNet(Layer):
    """
    组合计算层构成神经网络
    """

    def __init__(self, layers: Sequence[Layer]) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """
        前向传播,计算结果
        """
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        """
        反向传播,计算梯度
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterable[Tuple[Tensor, Tensor]]:
        """
        获取每一层的权重和对应的梯度,用于给优化器更新权重
        """
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
