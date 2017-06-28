# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.dataset import convert

#import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')    
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    flag_train = False if args.train < 0 else True
    flag_resum = False if args.use < 0 else True
    n_epoch = args.epoch if flag_train == True else 1
    
    tsm = MTModel(args.gpu, flag_train, flag_resum, n_epoch, args.batchsize)
    tsm.run()

class MTNNet(chainer.Chain):
    def __init__(self, n_mid, n_out):
        super(MTNNet, self).__init__()
        with self.init_scope():
            self.lin1 = L.Linear(None, n_mid)
            self.lin2 = L.Linear(None, n_out)
        
    def __call__(self, x):
        h1 = self.lin1(x)
        h2 = F.relu(h1)
        h3 = F.dropout(h2)
        
        y = self.lin2(h3)
        return y
    
    def loss(self, x, t):
        y = self(x)
        loss = F.softmax_cross_entropy(y,t)
        return loss, y

class MTModel():
    def __init__(self, gpu, flag_train, flag_resum, n_epoch, batchsize):
        self.n_epoch = n_epoch
        self.flag_train = flag_train
        self.flag_resum = flag_resum
        self.gpu = gpu
        
        self.model = MTNNet(256, 10)
        
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()
        
        if self.flag_train:
            self.optimizer = chainer.optimizers.Adam()
            self.optimizer.setup(self.model)
        
        if self.flag_resum:
            chainer.serializers.load_npz("./net/net.model", self.model)
        
        train, test = chainer.datasets.get_mnist()
        
        self.N_train = len(train)
        self.N_test = len(test)
        
        self.train_iter = chainer.iterators.SerialIterator(train, batchsize,
                                                           repeat=True, shuffle=True)
        self.test_iter = chainer.iterators.SerialIterator(test, self.N_test,
                                                          repeat=False, shuffle=False)
        
    def run(self):
        #xp = np if self.gpu < 0 else chainer.cuda.cupy
        sum_accuracy = 0
        sum_loss = 0
        
        while self.train_iter.epoch < self.n_epoch:
            batch = self.train_iter.next()
            x_array, t_array = convert.concat_examples(batch, self.gpu)
            x = chainer.Variable(x_array)
            t = chainer.Variable(t_array)
            optimizer.update(model, x, t)
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
            
            if train_iter.is_new_epoch:
                print('epoch: ', train_iter.epoch)
                print('train mean loss: {}, accuracy: {}'.format(
                    sum_loss / train_count, sum_accuracy / train_count))
                # evaluation
                sum_accuracy = 0
                sum_loss = 0
                for batch in test_iter:
                    x_array, t_array = convert.concat_examples(batch, args.gpu)
                    x = chainer.Variable(x_array)
                    t = chainer.Variable(t_array)
                    loss = model(x, t)
                    sum_loss += float(loss.data) * len(t.data)
                    sum_accuracy += float(model.accuracy.data) * len(t.data)
    
                test_iter.reset()
                print('test mean  loss: {}, accuracy: {}'.format(
                    sum_loss / test_count, sum_accuracy / test_count))
                sum_accuracy = 0
                sum_loss = 0
                
        chainer.serializers.save_npz('mlp.model', self.model)
        chainer.serializers.save_npz('mlp.state', self.optimizer)

if __name__ == '__main__':
    main()