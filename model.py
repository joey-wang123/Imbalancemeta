import torch.nn as nn
import torch

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv3x3act(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2)
    )

def conv3x3nopool(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def conv3x3nobatch(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )



class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks

        self.conv1 = conv3x3nobatch(in_channels, hidden_size)
        self.conv2 = conv3x3nobatch(hidden_size, hidden_size)
        self.conv3 = conv3x3nobatch(hidden_size, hidden_size)
        #self.conv3 = conv3x3(hidden_size, hidden_size)
        #self.conv4 = conv3x3(hidden_size, out_channels)
        self.domain_out = torch.nn.ModuleList()
        for _ in range(self.taskcla):
            self.task = nn.Sequential(
                conv3x3(hidden_size, hidden_size),
                conv3x3(hidden_size, out_channels)
            )
            self.domain_out.append(self.task)
        
        
    def forward(self, inputs, domain_id, s=1):
        #print('multi forward inference')
        h = self.conv1(inputs.view(-1, *inputs.shape[2:]))
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.domain_out[domain_id](h)
        return h.view(*inputs.shape[:2], -1)

  
    def set_req_grad(self, domain_id, req_grad):

        for i in range(self.taskcla):
            if i!= domain_id:
                params = list(self.domain_out[i].parameters()) 
                for ind in range(len(params)):
                    params[ind].requires_grad = req_grad
            else:
                params = list(self.domain_out[domain_id].parameters()) 
                for ind in range(len(params)):
                    params[ind].requires_grad = True
        return


class PrototypicalNetworkJoint(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0):
        super(PrototypicalNetworkJoint, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks

        '''
        self.conv1 = conv3x3(in_channels, hidden_size)
        self.conv2 = conv3x3(hidden_size, hidden_size)
        self.conv3 = conv3x3(hidden_size, hidden_size)
        self.conv4 = conv3x3(hidden_size, hidden_size)
        self.conv5 = conv3x3act(hidden_size, out_channels)
        #self.conv5 = conv3x3(hidden_size, out_channels)
        '''


        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3act(hidden_size, out_channels)
        )

        self.activation = nn.ReLU()
        #self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, inputs, domain_id = None, train = True):
        #print('multi forward inference')
        '''
        h = self.conv1(inputs.view(-1, *inputs.shape[2:]))
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        out = self.conv5(h)
        out.retain_grad()
        print('retain gradient')
        out2 = self.activation(out)
        '''

        out = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        '''
        if train:
            out.retain_grad()
        '''
        out2 = self.activation(out)
        return out2.view(*inputs.shape[:2], -1), out

  
