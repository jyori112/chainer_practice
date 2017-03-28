import numpy as np
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

import net

def main():
  model = L.Classifier(net.MNIST((28, 28, 1), (32, 5, 3), (32, 5, 2)))

  optimizer = chainer.optimizers.Adam()
  optimizer.setup(model)

  train, test = chainer.datasets.get_mnist(ndim=3)
  train_iter = chainer.iterators.SerialIterator(train, batch_size=100)
  test_iter = chainer.iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

  updater = training.StandardUpdater(train_iter, optimizer)
  trainer = training.Trainer(updater, (5, 'epoch'), out='result')

  trainer.extend(extensions.Evaluator(test_iter, model))
  trainer.extend(extensions.LogReport())
  trainer.extend(extensions.PrintReport(
      ['epoch', 'main/loss', 'validation/main/loss',
       'main/accuracy','validation/main/accuracy']))
  trainer.extend(extensions.ProgressBar())

  trainer.run()

if __name__ == '__main__':
    main()