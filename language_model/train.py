import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np

class LanguageModel(chainer.Chain):
  def __init__(self, n_vocab, train=True):
    super(LanguageModel, self).__init__(
        embed=L.EmbedID(n_vocab, 500),
        l1=L.LSTM(500, 500),
        l2=L.LSTM(500, 500),
        l3=L.Linear(500, n_vocab)
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
    self.train = train

  def reset_state(self):
    self.l1.reset_state()
    self.l2.reset_state()

  def __call__(self, x):
    h=self.embed(x)
    h=F.relu(self.l1(F.dropout(h, train=self.train)))
    h=F.relu(self.l2(F.dropout(h, train=self.train)))
    return self.l3(F.dropout(h, train=self.train))

class Iterator(chainer.dataset.Iterator):
  def __init__(self, dataset, batch_size, repeat=True):
    # Keep data
    self.dataset = dataset
    self.N = len(dataset)
    self.batch_size = batch_size
    self.repeat = repeat

    self.epoch = 0
    self.is_new_epoch = False

    self.offsets = np.asarray([i * self.N // batch_size for i in range(batch_size)], dtype=np.int32)

    self.iteration = 0

  def __next__(self):
    length = len(self.dataset)
    if not self.repeat and self.iteration * self.batch_size >= length:
      raise StopIteration

    x = self.dataset[(self.offsets+self.iteration) % self.N]
    t = self.dataset[(self.offsets+self.iteration+1) % self.N]

    self.iteration += 1

    epoch = self.iteration * self.batch_size // self.N
    self.is_new_epoch = self.epoch < epoch
    if self.is_new_epoch:
      self.epoch = epoch

    return list(zip(x, t))

  @property
  def epoch_detail(self):
    return self.iteration * self.batch_size / len(self.dataset)

  def serialize(self, serializer):
    self.iteration = serializer('iteration', self.iteration)
    self.epoch = serializer('epoch', self.epoch)

class BPTTUpdater(training.StandardUpdater):
  def __init__(self, train_iter, optimizer, bprop_len, device):
    super(BPTTUpdater, self).__init__(train_iter, optimizer, device=device)
    self.bprop_len = bprop_len

  def update_core(self):
    loss = 0

    train_iter = self.get_iterator('main')
    optimizer = self.get_optimizer('main')

    for i in range(self.bprop_len):
      batch = train_iter.__next__()

      x, t = self.converter(batch, self.device)

      loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

    optimizer.target.cleargrads()
    loss.backward()
    loss.unchain_backward()
    optimizer.update()

def main():
  train, val, test = chainer.datasets.get_ptb_words()
  n_vocab = max(train) + 1  # train is just an array of integers
  print('#vocab =', n_vocab)

  train = train[:100]
  val = val[:100]
  test = test[:100]

  train_iter = Iterator(train, 20)
  val_iter = Iterator(val, 1, repeat=False)
  test_iter = Iterator(test, 1, repeat=False)

  model = L.Classifier(LanguageModel(n_vocab))

  optimizer = chainer.optimizers.Adam()
  optimizer.setup(model)

  updater = BPTTUpdater(train_iter, optimizer, 20, -1)
  trainer = training.Trainer(updater, (20, 'epoch'), out="result")

  eval_model = model.copy()
  eval_rnn = eval_model.predictor
  eval_rnn.train = False

  trainer.extend(extensions.Evaluator(
        val_iter, eval_model,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))

  trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
  trainer.extend(extensions.snapshot())
  trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'))
  trainer.extend(extensions.LogReport())
  trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
  trainer.extend(extensions.ProgressBar())
  trainer.run()

  print('test')
  eval_rnn.reset_state()
  evaluator = extensions.Evaluator(test_iter, eval_model)
  result = evaluator()
  print('test perplexity:', np.exp(float(result['main/loss'])))

if __name__ == '__main__':
    main()