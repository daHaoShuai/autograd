try:
    import urllib.request
except ImportError:
    raise ImportError('需要使用 Python 3.x')
import gzip
import pickle
import os
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


class Mnist:
    """
    mnist数据集
    """

    def __init__(self, save_file: str = '/mnist.pkl') -> None:
        self.url_base = 'http://yann.lecun.com/exdb/mnist/'
        self.key_file = {
            'train_img': 'train-images-idx3-ubyte.gz',
            'train_label': 'train-labels-idx1-ubyte.gz',
            'test_img': 't10k-images-idx3-ubyte.gz',
            'test_label': 't10k-labels-idx1-ubyte.gz'
        }
        self.dataset_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_file = self.dataset_dir + save_file

    def _download_mnist(self):
        """
        下载mnist数据集
        """
        for file_name in self.key_file.values():
            file_path = self.dataset_dir + "/" + file_name
            if os.path.exists(file_path):
                return
            print("正在下载 " + file_name + " ... ")
            urllib.request.urlretrieve(self.url_base + file_name, file_path)
            print("下载完成")

    def _load_label(self, file_name):
        """
        转换标签
        """
        file_path = self.dataset_dir + "/" + file_name
        print("正在转换 " + file_name + " 为 NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print("完成")
        return labels

    def _load_img(self, file_name):
        """
        转换图片
        """
        file_path = self.dataset_dir + "/" + file_name
        print("正在转换 " + file_name + " 为 NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 784)  # 28 * 28 = 784
        print("完成")
        return data

    def _convert_numpy(self):
        """
        转换成numpy数组
        """
        dataset = {}
        dataset['train_img'] = self._load_img(self.key_file['train_img'])
        dataset['train_label'] = self._load_label(self.key_file['train_label'])
        dataset['test_img'] = self._load_img(self.key_file['test_img'])
        dataset['test_label'] = self._load_label(self.key_file['test_label'])
        return dataset

    def init_mnist(self):
        """
        下载并创建pickle文件
        """
        self._download_mnist()
        dataset = self._convert_numpy()
        print("正在创建 pickle 文件 ...")
        with open(self.save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("完成")

    def _change_one_hot_label(self, X):
        """
        转换成ont_hot
        """
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
        return T

    def load(self, normalize=True, flatten=True, one_hot_label=False):
        """读入MNIST数据集
        Parameters
        ----------
        normalize : 将图像的像素值正规化为0.0~1.0
        one_hot_label : 
            one_hot_label为True的情况下,标签作为one-hot数组返回
            one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
        flatten : 是否将图像展开为一维数组

        Returns
        -------
        (训练图像, 训练标签), (测试图像, 测试标签)
        """
        if not os.path.exists(self.save_file):
            self.init_mnist()

        with open(self.save_file, 'rb') as f:
            dataset = pickle.load(f)

        if normalize:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0

        if one_hot_label:
            dataset['train_label'] = self._change_one_hot_label(
                dataset['train_label'])
            dataset['test_label'] = self._change_one_hot_label(
                dataset['test_label'])

        if not flatten:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

        return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

