from typing import List
import numpy as np
from nn.nn import NeuralNet
from nn.layers import Linear, Tanh
from nn.optim import Adam
from nn.loss import CrossEntropyLoss
from nn.train import train


def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]


def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def predict(net: NeuralNet) -> None:
    """
    验证准确率
    """
    num_correct = 0
    for x in range(1, 101):
        predicted = net(binary_encode(x))
        predicted_idx = np.argmax(predicted)
        actual_idx = np.argmax(fizz_buzz_encode(x))
        labels = [str(x), 'fizz', 'buzz', 'fizzbuzz']
        if predicted_idx == actual_idx:
            num_correct += 1
        print(
            f'x : {x} | 预测值 : {labels[predicted_idx]} | 正确值 : {labels[actual_idx]}')
    print(num_correct, '/ 100')


inputs = np.array(
    [binary_encode(x) for x in range(101, 1024)]
)

targets = np.array(
    [fizz_buzz_encode(x) for x in range(101, 1024)]
)

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])
# 训练
train(net, inputs, targets, num_epochs=2000,
      optimizer=Adam(), loss=CrossEntropyLoss())
# 预测
predict(net)
