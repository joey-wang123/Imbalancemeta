import numpy as np
import torch
import torch.nn as nn
import os

from net.convnet import ConvNet
from net.resnet import ResNet

class GBML:
    '''
    Gradient-Based Meta-Learning
    '''
    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.batch_size
        return None

    def _init_net(self):
        if self.args.net == 'ConvNet':
            self.network = ConvNet(self.args)
        elif self.args.net == 'ResNet':
            self.network = ResNet(self.args)
            self.args.hidden_channels = 640
        self.network.train()
        self.network.cuda()
        return None

    def _init_opt(self):

        '''
        inner_parameters = []
        for name, p in self.network.named_parameters():
            if p.requires_grad:
                if 'encoder' not in name:
                    print('name', name)
                    inner_parameters.append(p)
        '''

        '''
        if self.args.inner_opt == 'SGD':
            self.inner_optimizer = torch.optim.SGD(inner_parameters, lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer = torch.optim.Adam(inner_parameters, lr=self.args.inner_lr, betas=(0.0, 0.9))
        else:
            raise ValueError('Not supported inner optimizer.')
        '''
        
        if self.args.inner_opt == 'SGD':
            self.inner_optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.inner_lr, betas=(0.0, 0.9))
        else:
            raise ValueError('Not supported inner optimizer.')
        
        inner_parameters = []
        for name, p in self.network.named_parameters():
            if p.requires_grad:
                if 'encoder' not in name:
                    print('inner optimizer name', name)
                    inner_parameters.append(p)
        self.inner_optimizer = torch.optim.SGD(inner_parameters, lr=self.args.inner_lr)
        
        if self.args.outer_opt == 'SGD':
            self.outer_optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.outer_lr, nesterov=True, momentum=0.9)
        elif self.args.outer_opt == 'Adam':
            self.outer_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.outer_lr)
        else:
            raise ValueError('Not supported outer optimizer.')
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=10, gamma=0.5)
        return None

    def unpack_batch(self, batch):
        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.cuda()
        train_targets = train_targets.cuda()

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.cuda()
        test_targets = test_targets.cuda()
        return train_inputs, train_targets, test_inputs, test_targets

    def inner_loop(self):
        raise NotImplementedError

    def outer_loop(self):
        raise NotImplementedError

    def lr_sched(self):
        self.lr_scheduler.step()
        return None

    def load(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.load_path)
        self.network.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.load_path)
        self.network.encoder.load_state_dict(torch.load(path))

    def save(self, epoch, shots):
        path = os.path.join(self.args.result_path, self.args.alg, 'shots'+str(shots))
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'epoch'+str(epoch)+'.pth')
        torch.save(self.network.state_dict(), save_path)

