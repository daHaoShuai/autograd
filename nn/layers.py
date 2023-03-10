from typing import Any, Dict, Callable
import numpy as np
from nn.tensor import Tensor
from nn.function import sigmoid, tanh, tanh_grad, relu, relu_grad


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return self.forward(*args, **kwds)


class Linear(Layer):
    """
    全连接层 : w * x + b
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
             dy/db = f'(x) * a
             dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
             dy/db = a.T @ f'(x)
             dy/bc = f'(x)
        """
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


class Tanh(Activation):
    """
    tanh激活函数
    """

    def __init__(self) -> None:
        super().__init__(tanh, tanh_grad)


class Relu(Activation):
    """
    relu激活函数
    """

    def __init__(self) -> None:
        super().__init__(relu, relu_grad)


class Sigmoid(Activation):
    """
    sigmoid函数
    """

    def __init__(self) -> None:
        self.out = None

        def sigmoid_forward(x: Tensor) -> Tensor:
            out = sigmoid(x)
            self.out = out
            return out

        def sigmoid_prime(x: Tensor) -> Tensor:
            return x * (1.0 - self.out) * self.out

        super().__init__(sigmoid_forward, sigmoid_prime)
