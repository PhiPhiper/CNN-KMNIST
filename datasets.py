##################################################
## Solving the KMNIST Classifation task with smac
## to do some execise
##################################################
## 
##################################################
## Author: Philipp Jankov
## Copyright: Copyright 2019, Philipp Jankov
## Credits: KMNIST class was kindly provided
##          by Marius Lindauer
##################################################

import os

import requests
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from collections import deque
from typing import Iterable, List, Optional, Tuple, Callable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from apex import amp

import matplotlib.pyplot as plt
import torchvision


class KMNIST(Dataset):
    """
    Dataset class for use with pytorch for the Kuzushiji-MNIST dataset as given in
    Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. arXiv:1812.01718

    Kuzushiji-MNIST contains 70,000 28x28 grayscale images spanning 10 classes (one from each column of hiragana),
    and is perfectly balanced like the original MNIST dataset (6k/1k train/test for each class).
    """

    def __init__(self, data_dir='.', train: bool = True, transform=None):
        """
        :param data_dir: Directory of the data
        :param train: Use training or test set
        :param transform: pytorch transforms for data augmentation
        """

        self.__urls = [
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz',
        ]

        t_str = 'train' if train else 'test'
        imgs_fn = 'kmnist-{}-imgs.npz'.format(t_str)
        labels_fn = 'kmnist-{}-labels.npz'.format(t_str)

        if not os.path.exists(data_dir):
            os.mkdir(os.path.abspath(data_dir))
        if not os.path.exists(os.path.abspath(os.path.join(data_dir, imgs_fn))):
            self.__download(os.path.abspath(data_dir))

        imgs_fn = os.path.abspath(os.path.join(data_dir, imgs_fn))
        labels_fn = os.path.abspath(os.path.join(data_dir, labels_fn))

        self.images = np.load(imgs_fn)['arr_0']
        self.labels = np.load(labels_fn)['arr_0']
        self.images_tensor = torch.unsqueeze(torch.Tensor(self.images / 255), 1) # normalize image data
        self.labels_tensor = torch.Tensor(self.labels).type(torch.LongTensor)
        self.n_classes = len(np.unique(self.labels))
        self.class_labels, self.class_frequency = np.unique(self.labels, return_counts=True)
        self.class_frequency = self.class_frequency / np.sum(self.class_frequency)
        self.data_dir = data_dir
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1  # only gray scale
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images_tensor[idx]
        if self.transform:
            image = self.transform(image)

        label = self.labels_tensor[idx]
        return image, label

    def __download(self, data_dir):
        print('Datadir', data_dir)
        for url in self.__urls:
            fn = os.path.basename(url)
            req = requests.get(url, stream=True)
            print('Downloading {}'.format(fn))
            with open(os.path.join(data_dir, fn), 'wb') as fh:
                for chunck in req.iter_content(chunk_size=1024):
                    if chunck:
                        fh.write(chunck)
            print('done')
        print('All files downloaded')

    def show_samples(self):
        randidx = np.random.random_integers(0, len(self.labels), 64)
        images, labels = self.images_tensor[randidx], self.labels[randidx]
        grid = torchvision.utils.make_grid(images * 255)

        plt.imshow((grid.numpy().transpose((1, 2, 0))))
        plt.axis('off')
        plt.title(labels)
        plt.show()

class Flatten(nn.Module):
    """ Flatten the input to (batchsize, feature_dim)
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x.shape
        return x.reshape(x.shape[0], -1)
    
    # def backward(self, grad: np.ndarray) -> np.ndarray:
    #     old_shape = self.input_cache
    #     return grad.reshape(old_shape)

class MyCNN(nn.Module):
    """ Simple CNN with 3 conv layers and a linear output layer
    """
    def __init__(self, in_shape: Tuple[int], in_channels: int, out_channels: int, kernel_shape: Tuple[int] = (3, 3),
                 stride: Tuple[int] = (1, 1), dropout: float = .0):
        super().__init__()
        self.in_shape = in_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.dropout = dropout
        num_filter_c0 = 64
        num_filter_c1 = 128
        num_filter_c2 = 256

        # Architecture
        self.net = nn.Sequential()
        outshape, padding = self.get_outshape_and_padding(self.in_shape)
        self.net.add_module('conv_0', nn.Conv2d(in_channels, num_filter_c0, self.kernel_shape, stride=self.stride, padding=padding))
        self.net.add_module('relu_0', nn.ReLU())
        self.net.add_module('dropout_0', nn.Dropout2d(self.dropout))
        outshape, padding = self.get_outshape_and_padding(outshape)
        self.net.add_module('conv_1', nn.Conv2d(num_filter_c0, num_filter_c1, self.kernel_shape, stride=self.stride, padding=padding))
        self.net.add_module('relu_1', nn.ReLU())
        self.net.add_module('dropout_1', nn.Dropout2d(self.dropout))
        outshape, padding = self.get_outshape_and_padding(outshape)
        self.net.add_module('conv_2', nn.Conv2d(num_filter_c1, num_filter_c2, self.kernel_shape, stride=self.stride, padding=padding))
        self.net.add_module('relu_2', nn.ReLU())
        self.net.add_module('dropout_2', nn.Dropout2d(self.dropout))

        self.net.add_module('flat', Flatten())
        
        self.net.add_module('lin_out', nn.Linear(np.prod(outshape) * num_filter_c2, self.out_channels))

        # Weight and Bias Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.net(x)

    def get_outshape_and_padding(self, in_shape: Tuple[int]) -> (Tuple[int], Tuple[int]):
        in_shape = np.array(in_shape)
        kernel_shape = np.array(self.kernel_shape)
        stride = np.array(self.stride)
        padding = np.array((0, 0))

        out_shape = ((in_shape - kernel_shape + 2 * padding) / stride) + 1

        # silent padding
        if np.any(out_shape % 1 > 1):
            padding = out_shape % 1 * stride
            out_shape = ((in_shape - kernel_shape + 2 * padding) / stride) + 1
        return (tuple(out_shape.astype(int).tolist()), tuple(padding.astype(int).tolist()))

def split_ratio_to_lengths(quota: int, ratiolist: Iterable) -> List[int]:
    """ Splits a given length into lengths according to the given ratiolist
    """
    ratios = np.array(ratiolist)
    ratios = ratios / np.sum(ratios) # normalize just in case
    dsetsizes = []
    restquota = quota
    for ratio in ratios.tolist():
        size = int(np.round(quota * ratio))
        assert size <= restquota
        dsetsizes.append(size)
        restquota -= size
    return dsetsizes

def accuracy(y_predicted, y_true, y_true_is_onehot: bool = False) -> float:
    """ Assume both are one hot encoded
    """
    y_predicted = np.argmax(np.array(y_predicted).reshape(-1, 10), axis=-1)
    y_true = np.argmax(np.array(y_true).reshape(-1, 10), axis=-1) if y_true_is_onehot \
             else np.array(y_true).reshape(-1)
    return np.sum(np.equal(y_true, y_predicted)) / len(y_true)

def train(model, loss_fn, optimizer, trainset, valset, batchsize, mp=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(trainset, batch_size=batchsize, pin_memory=True, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, pin_memory=True, shuffle=True)

    # Preallocate space
    train_loss, train_accuracies = np.zeros(1000), np.zeros(1000)
    val_loss, val_accuracies = np.zeros(1000), np.zeros(1000)

    training_true, training_predictions = [], []
    val_true, val_predictions = [], []

    # Use exponential moving average over the validation accuracy to earlystop learning
    valacc_ema_q = deque(maxlen = 10)
    alpha = 0.9
    epoch = 0
    while(True):
        start = time.time()
        epoch += 1
        print("Epoch {}".format(epoch))
        if epoch > train_loss.shape[0]:
            train_loss = np.append(train_loss, np.zeros(1000))
            train_accuracies = np.append(train_accuracies, np.zeros(1000))
            val_loss = np.append(val_loss, np.zeros(1000))
            val_accuracies = np.append(val_accuracies, np.zeros(1000))

        # Train
        training_true.clear()
        training_predictions.clear()
        model.train()
        for x_batch, y_batch in iter(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()

            y_batch_predicted = model(x_batch)
            loss = loss_fn(y_batch_predicted, y_batch)
            if mp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            train_loss[epoch - 1] += float(loss)
            training_true.append(np.array(y_batch.cpu()))
            training_predictions.append(np.array(y_batch_predicted.data.cpu()))
        train_accuracies[epoch - 1] = accuracy(training_predictions, training_true)


        # Validate
        val_true.clear()
        val_predictions.clear()
        model.eval()
        for x_batch, y_batch in iter(val_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_batch_predicted = model(x_batch)
            loss = loss_fn(y_batch_predicted, y_batch)

            val_loss[epoch - 1] += float(loss)
            val_true.append(np.array(y_batch.cpu()))
            val_predictions.append(np.array(y_batch_predicted.data.cpu()))
        val_accuracies[epoch - 1] = accuracy(val_predictions, val_true)

        print("  Training Loss:\t{:.4f}".format(train_loss[epoch - 1]))
        print("  Validation Loss:\t{:.4f}".format(val_loss[epoch - 1]))
        print("  Training Accuracy:\t{:.4f}".format(train_accuracies[epoch - 1]))
        print("  Validation Accuracy:\t{:.4f}".format(val_accuracies[epoch - 1]))
 
        # Calc exponential moving average for the current epoch
        valacc_ema_q.appendleft(valacc_ema_q[0] + alpha * (val_accuracies[epoch - 1] - valacc_ema_q[0]) if epoch > 1 else val_accuracies[epoch - 1])
        print("  Validation Loss (EMA):\t{:.4f}".format(valacc_ema_q[0]))
        print("  This epoch took\t{:.4f}s".format(time.time()-start))
        if epoch > valacc_ema_q.maxlen and valacc_ema_q[0] - valacc_ema_q[-1] < 5.*1e-3:
           break

    return train_loss[:epoch], train_accuracies[:epoch], val_loss[:epoch], val_accuracies[:epoch]

class smaced():
    def __init__(self):
        self.min_validationerror = float('Inf')
        self.saved_model = None
        self.history = None
        self.criterion = nn.CrossEntropyLoss()
        self.test_score = None

    def run_smac_config(self, cfg):
        return self.run_config(**cfg)

    def run_config(self, lr, dropout_p, mp=False):
        # Define dataset
        kmnist = KMNIST()
        # Define the network params
        filter_shape = (4, 4)
        stride = (1, 1)
        myCNN = MyCNN((kmnist.img_cols, kmnist.img_rows), kmnist.channels, kmnist.n_classes, filter_shape, stride, dropout_p)
        if torch.cuda.is_available(): myCNN.cuda()
        # Define loss and optimizer
        optimizer = optim.Adam(myCNN.parameters(), lr=lr)

        if mp:
            myCNN, optimizer = amp.initialize(myCNN, optimizer, opt_level="O1")

        # Split data
        trainset, valset, testset = torch.utils.data.random_split(kmnist, split_ratio_to_lengths(len(kmnist), (.7, .15, .15)))
        tl, ta, vl, va = train(myCNN, self.criterion, optimizer, trainset, valset, 500, mp) # <- 1000 was too big for my 8GB GPU
        if (1-va[-1]) < self.min_validationerror:
            self.min_validationerror = (1-va[-1])
            self.saved_model = myCNN
            self.history = np.vstack((tl, ta, vl, va)).T
            self.test_score = (self.test(testset))
        return (1-va[-1])

    def test(self, dset):
        test_loader = DataLoader(dset, batch_size=500, pin_memory=True, shuffle=True)
        test_true, test_predictions = [], []
        test_loss = 0
        test_accuracies = 0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.saved_model.eval()
        for x_batch, y_batch in iter(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_batch_predicted = self.saved_model(x_batch)
            loss = self.criterion(y_batch_predicted, y_batch)

            test_loss += float(loss)
            test_true.append(np.array(y_batch.cpu()))
            test_predictions.append(np.array(y_batch_predicted.data.cpu()))
        test_accuracies = accuracy(test_predictions, test_true)
        print("##############################################")
        print("Test Loss:\t{:.4f}".format(test_loss))
        print("Test Accuracy:\t{:.4f}".format(test_accuracies))
        print("##############################################")
        return test_loss, test_accuracies

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    kmnist = KMNIST(train=False)
    s = smaced()
    s.run_config(0.001, 0.1)

    s = smaced()
    s.run_config(0.001, 0.1, mp=True)
