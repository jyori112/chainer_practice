import numpy as np
import chainer

import chainer.functions as F
import chainer.links as L

class MNIST(chainer.Chain):
  def __init__(self, in_dim, conv1, conv2):
    self.h, self.w, self.ch = in_dim
    self.ch1, self.ksize1, self.pool1 = conv1
    self.ch2, self.ksize2, self.pool2 = conv2

    super(MNIST, self).__init__(
      conv1 = L.Convolution2D(1, self.ch1, self.ksize1, 1, 1),
      conv2 = L.Convolution2D(self.ch1, self.ch2, self.ksize2, 1, 1),
      l1 = L.Linear(self.ch2 * (self.h // (self.pool1 * self.pool2)) * (self.w // (self.pool1 * self.pool2)), 10)
    )

  def __call__(self, x):
    h1 = F.max_pooling_2d(F.relu(self.conv1(x)), self.pool1)
    h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), self.pool2)
    return self.l1(h2)
