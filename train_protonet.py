
import torch
from tqdm import tqdm
import logging
from torchmeta.utils.prototype import get_prototypes, prototypical_loss

from model import PrototypicalNetworkJoint
from utils import get_accuracy, gauss_kernel
import numpy as np
from datasets import *
import pickle
import random
import copy
from torch.distributions.categorical import Categorical
import domain_shift as algos
import warnings
warnings.filterwarnings("ignore")

datanames = ['Quickdraw', 'MiniImagenet', 'Omniglot', 'CUB', 'Aircraft', 'Necessities']

class SequentialMeta(object):
    def __init__(self,model, optimizer, args=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.step = 0
        self.memory_rep = []
        str_save = '_'.join(datanames)
        self.data_step = {}
        self.domain_id  = 0
        self.memory_step = 0
        self.domain_iter = {}
        for ind in range(30):
            self.domain_iter[str(ind)] = 0
        self.domain_embed = 0
        self.memory_rep = []
        self.window = []

        for name in datanames:
            self.data_step[name] = 0

        self.memory_count = {}
        for ind in range(len(datanames)+5):
            self.memory_count[ind] = 0

        self.memory_cluster = {}
        for ind in range(10):
            self.memory_cluster[ind] = []


        self.softmax = torch.nn.Softmax(dim=0)
        self.startiter = 0
        self.numsteps = 3
        self.Reservoir = False
        self.detect_shift = True
        self.Reservoir = args.Reservior
        print('self.Reservoir', self.Reservoir)
        if self.Reservoir:
            savemode = 'Reservoir'
        else:
            savemode = 'Balance'

        B = 200
        big_Lambda, small_lambda = algos.select_optimal_parameters(B)
        thres_ff = small_lambda
        sigmasq = 0.02
        X0 = np.array([0.21, 0.12, 0.15])
        N = 3
        self.detector = algos.Online_domainshift_detection(X0, kernel_func=lambda x, y:gauss_kernel(x, y, np.sqrt(sigmasq)), window_size=B,
                               nbr_windows=N, adapt_forget_factor=thres_ff)

        self.filepath = os.path.join(self.args.output_folder, 'protonet_{}_{}_Memory_{}'.format(savemode, str_save, self.args.memory_limit),  'shot{}'.format(self.args.train_shot), 'way{}'.format(self.args.num_way))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

    def train(self, dataloader_dict, epoch):


        self.model.train()
        for dataname, dataloader in dataloader_dict.items():
            with tqdm(dataloader, total=self.args.num_batches) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    self.model.zero_grad()
                    train_inputs, train_targets = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    train_targets = train_targets.to(device=self.args.device)
                    if train_inputs.size(2) == 1:
                        train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                    train_embeddings, pre_acttrain = self.model(train_inputs)

                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    if test_inputs.size(2) == 1:
                        test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                    test_embeddings, pre_acttest = self.model(test_inputs)

                    prototypes = get_prototypes(train_embeddings, train_targets, self.args.num_way)
                    loss = prototypical_loss(prototypes, test_embeddings, test_targets)
                    current = copy.deepcopy(loss.detach())

                    #domain shift detection
                    if self.detect_shift:
                        alpha = 0.5
                        mean_proto = torch.mean(prototypes, dim = (0,1))
                        self.domain_embed = alpha*torch.mean(test_embeddings, dim= (0,1)) + (1-alpha)*self.domain_embed

                        lenwindow = 20
                        if (len(self.window))> lenwindow:
                            self.window.remove(self.window[0])
                        
                        if self.window:
                            if (len(self.window)) == lenwindow:
                                dist_list = []
                                for proto in self.window[-1*(self.numsteps+1):-1]:
                                    currentdist = torch.sum((mean_proto - proto)**2).item()
                                    dist_list.append(currentdist)


                            if self.step>lenwindow:

                                x = torch.tensor(dist_list).cpu().detach().numpy()
                                self.detector.update(x)
                                result = self.detector.flag_sample()
                                if result and self.domain_iter[str(self.domain_id)]>600:
                                    self.domain_id = self.domain_id + 1
                                self.domain_iter[str(self.domain_id)]+= 1
                        self.window.append(self.domain_embed) 

                        if self.domain_id >= (len(datanames)-1):
                            self.detect_shift = False



                    if self.memory_rep:
                        if self.Reservoir:
                            memory_loss = self.Reservoir_memory2()
                        else:
                            memory_loss = self.Reservoir_memory()
                        loss += memory_loss


                    #Reservoir sampling
                    if self.Reservoir:
                        if self.step < self.args.memory_limit:
                            savedict = {self.domain_id: batch}
                            self.memory_rep.append(batch)

                        else:
                            randind = random.randint(0, self.step)
                            if randind < self.args.memory_limit:
                                savedict = {self.domain_id: batch}
                                self.memory_rep[randind] = batch

                    else:

                        eachsize = self.args.memory_limit
                        #sample in
                        if len(self.memory_rep)<eachsize:
                            savedict =  batch
                            self.memory_cluster[self.domain_id].append(savedict)
                            self.memory_rep.append(savedict)

                        self.memory_count[self.domain_id] = self.memory_count[self.domain_id] +1

                        if self.domain_id>0:
                            ratio = len(self.memory_cluster[self.domain_id])/float(self.args.memory_limit)
                            score_memory = ratio*(memory_loss/self.args.sample)
                            beta = 1.0
                            factor = torch.exp(torch.tensor([-beta*self.memory_count[self.domain_id]])).to(self.args.device)
                            score_current = factor*(1-ratio)*current  

                            if len(self.memory_cluster[self.domain_id])>self.args.memory_limit/(self.domain_id+1):
                                score_memory += 2.0

                            score_prob = self.softmax(50.0*torch.tensor([score_memory, score_current]))   

                            samplein = torch.bernoulli(score_prob[1])
                            if samplein:
                                if self.domain_id>1:
                                    indlist = []
                                    for ind in range (self.domain_id):
                                        num_cluster = len(self.memory_cluster[ind])
                                        if num_cluster >= self.args.memory_limit/(2*(self.domain_id+1)):
                                            indlist.append(ind)

                                    scoreout = torch.zeros(len(indlist))
                                    for i, ind in enumerate (indlist):
                                        num_cluster = len(self.memory_cluster[ind])
                                        scoreout[i] = -1.0*(1.0 - num_cluster/float(self.args.memory_limit))*self.score_cluster[ind]

                                    scoreout = self.softmax(scoreout)
                                    dist = Categorical(scoreout)
                                    sampleind = dist.sample().item()

                                    out_cluster = indlist[sampleind]
                                    self.memory_cluster[out_cluster].pop()
                                    self.memory_cluster[self.domain_id].append(batch)
                                else:
                                    self.memory_cluster[self.domain_id-1].pop()
                                    self.memory_cluster[self.domain_id].append(batch)

                            size_list = []
                            for ind in range(self.domain_id+1):
                                size_list.append(len(self.memory_cluster[ind]))

                    self.data_step[dataname] = self.data_step[dataname] + 1
                    self.step = self.step+1
                    

                    loss.backward()
                    self.optimizer.step()
                    if batch_idx >= self.args.num_batches:
                        break

    def Reservoir_memory(self):

        gradientnorm = False
        memory_loss =0
        count = self.args.sample
        num_memory = len(self.memory_rep)
        self.score_cluster = torch.zeros(self.domain_id)

        if num_memory<count:
            selectmemory = self.memory_rep
        else:
            current = self.memory_cluster[self.domain_id]
            if self.domain_id>1 and len(current)>1:
                count = 2
                model = copy.deepcopy(self.model)
                self.count_cluster = torch.zeros((self.domain_id))
                for ind in range(self.domain_id):
                    prev = self.memory_cluster[ind]
                    num_memory = len(prev)
                    self.count_cluster[ind] = num_memory
                    samplelist = random.sample(range(num_memory), count)
                    selectmemory = self.concatbatch(prev, samplelist)

                    if gradientnorm:
                        memoryloss, actgrad = self.memoryloss(selectmemory, model)
                        prevgradnorm = torch.norm(actgrad, p=2, dim = (1,2))
                        self.score_cluster[ind] = torch.mean(prevgradnorm)
                    else:
                        memoryloss, actgrad = self.memoryloss(selectmemory, model)
                        self.score_cluster[ind] = memoryloss.detach()

                num_memory = len(current)
                samplelist = random.sample(range(num_memory), count)
                selectmemory = self.concatbatch(current, samplelist)
                memoryloss, actgrad = self.memoryloss(selectmemory, model)
                currentgradnorm = torch.norm(actgrad, p=2, dim = (1,2))

                clusterlist = random.sample(range(self.domain_id), 2)
                selectmemory = []
                for ind in clusterlist:
                    selectmemory.append(random.choice(self.memory_cluster[ind]))  

            else:
                selectmemory = random.choice(self.memory_cluster[0]) 
                selectmemory = [selectmemory]
            for select in selectmemory:
                memory_loss += self.memoryloss(select)[0]

        return memory_loss

    def Reservoir_memory2(self):

        memory_loss =0
        num_memory = len(self.memory_rep)
        if num_memory<1:
            selectmemory = self.memory_rep
        else:
            samplelist = random.sample(range(num_memory), 1)
            selectmemory = []
            for ind in samplelist:
                selectmemory.append(self.memory_rep[ind])
        for select in selectmemory:

            memory_train_inputs, memory_train_targets = select['train'] 
            memory_train_inputs = memory_train_inputs.to(device=self.args.device)
            memory_train_targets = memory_train_targets.to(device=self.args.device)
            if memory_train_inputs.size(2) == 1:
                memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
            memory_train_embeddings, pre_acttrain = self.model(memory_train_inputs)

            memory_test_inputs, memory_test_targets = select['test'] 
            memory_test_inputs = memory_test_inputs.to(device=self.args.device)
            memory_test_targets = memory_test_targets.to(device=self.args.device)
            if memory_test_inputs.size(2) == 1:
                memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

            memory_test_embeddings, pre_acttest = self.model(memory_test_inputs)
            memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, self.args.num_way)
            memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)
        return memory_loss




    def concatbatch(self, prev, samplelist):
        selectmemory = {}
        traininputslist = []
        traintargetslist = []
        testinputslist = []
        testtargetslist = []
        for ind in samplelist:
            train_inputs, train_targets = prev[ind]['train']
            test_inputs, test_targets = prev[ind]['test']
            if train_inputs.size(2) == 1:
                train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
            if test_inputs.size(2) == 1:
                test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)

            traininputslist.append(train_inputs)
            traintargetslist.append(train_targets)
            testinputslist.append(test_inputs)
            testtargetslist.append(test_targets)

        stack_traininputs = torch.cat(traininputslist, dim = 0)
        stack_traintargets = torch.cat(traintargetslist, dim = 0)

        stack_testinputs = torch.cat(testinputslist, dim = 0)
        stack_testtargets = torch.cat(testtargetslist, dim = 0)

        selectmemory['train'] = [stack_traininputs, stack_traintargets]
        selectmemory['test'] = [stack_testinputs, stack_testtargets]


        return selectmemory

    def memoryloss(self, select, model = None):

        if model:
            newmodel  = model
        else:
            newmodel = self.model

        memory_train_inputs, memory_train_targets = select['train'] 
        memory_train_inputs = memory_train_inputs.to(device=self.args.device)
        memory_train_targets = memory_train_targets.to(device=self.args.device)
        if memory_train_inputs.size(2) == 1:
            memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
        memory_train_embeddings, pre_acttrain = newmodel(memory_train_inputs)

        memory_test_inputs, memory_test_targets = select['test'] 
        memory_test_inputs = memory_test_inputs.to(device=self.args.device)
        memory_test_targets = memory_test_targets.to(device=self.args.device)
        if memory_test_inputs.size(2) == 1:
            memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

        memory_test_embeddings, pre_acttest = newmodel(memory_test_inputs)
        memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, self.args.num_way)
        memory_loss = prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)

        if model:
            actgrad = torch.autograd.grad(memory_loss, pre_acttest)[0]
            actgrad = actgrad.view(*memory_test_inputs.shape[:2], -1)
            self.model.zero_grad()
        else:
            actgrad = None


        return memory_loss, actgrad

    def save(self, epoch):
        # Save model
        if self.args.output_folder is not None:
            filename = os.path.join(self.filepath, 'epoch{0}.pt'.format(epoch))
            with open(filename, 'wb') as f:
                state_dict = self.model.state_dict()
                torch.save(state_dict, f)

    def load(self, epoch):
        filename = os.path.join(self.filepath, 'epoch{0}.pt'.format(epoch))
        self.model.load_state_dict(torch.load(filename))


    def valid(self, epoch, dataloader_dict):
        self.model.eval()
        acc_list = []
        for dataname, dataloader in dataloader_dict.items():
            with torch.no_grad():
                with tqdm(dataloader, total=self.args.num_valid_batches) as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        self.model.zero_grad()

                        train_inputs, train_targets = batch['train']
                        train_inputs = train_inputs.to(device=self.args.device)
                        train_targets = train_targets.to(device=self.args.device)
                        if train_inputs.size(2) == 1:
                            train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                        train_embeddings, pre_acttrain = self.model(train_inputs, train = False)

                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=self.args.device)
                        test_targets = test_targets.to(device=self.args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                        test_embeddings, pre_acttest = self.model(test_inputs, train = False)

                        prototypes = get_prototypes(train_embeddings, train_targets, self.args.num_way)
                        accuracy, _, _ = get_accuracy(prototypes, test_embeddings, test_targets)
                        acc_list.append(accuracy.cpu().data.numpy())
                        pbar.set_description('dataname {} accuracy ={:.4f}'.format(dataname, np.mean(acc_list)))
                        if batch_idx >= self.args.num_valid_batches:
                            break

            avg_accuracy = np.round(np.mean(acc_list), 4)
            acc_dict = {dataname:avg_accuracy}
            return acc_dict

def main(args):

        all_accdict = {}
        train_loader_list, valid_loader_list, test_loader_list = dataset(args, datanames)
        model = PrototypicalNetworkJoint(3,
                                    args.embedding_size,
                                    hidden_size=args.hidden_size)
        model.to(device=args.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        seqmeta = SequentialMeta(model, optimizer, args=args)

        all_accdict['sample'] = args.sample
        all_accdict['memory_limit'] = args.memory_limit

        dataname = []
        total_epoch = 0
        for loaderindex, train_loader in enumerate(train_loader_list):
            each_epoch = args.epochs[loaderindex]

            for epoch in range(total_epoch, total_epoch+each_epoch):
                print('Epoch {}'.format(epoch))
                dataname.append(list(train_loader.keys())[0])
                
                seqmeta.train(train_loader, epoch)
                epoch_acc = []
                total_acc = 0.0
                for index, test_loader in enumerate(test_loader_list[:loaderindex+1]):
                    test_accuracy_dict = seqmeta.valid(epoch, test_loader)
                    epoch_acc.append(test_accuracy_dict)
                    acc = list(test_accuracy_dict.values())[0]
                    total_acc += acc
                avg_acc = total_acc/(len(test_loader_list[:loaderindex+1]))
                print('average testing accuracy', avg_acc)
                seqmeta.save(epoch)
                all_accdict[str(epoch)] = epoch_acc
                with open(seqmeta.filepath + '/stats_accnormal_combine.pickle', 'wb') as handle:
                    pickle.dump(all_accdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            total_epoch += each_epoch

        indexlist = []
        for memory in seqmeta.memory_rep:
            indexlist.append(list(memory))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')
    parser.add_argument('--data_path', type=str, default='meta_data/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--train-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5). during training')
    parser.add_argument('--test-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5) during testing.')
    parser.add_argument('--num-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output_folder', type=str, default='output/datasset/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks (default: 16).')


    parser.add_argument('--MiniImagenet_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for MiniImagenet (default: 4).')
    parser.add_argument('--CUB_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for CUB (default: 4).')
    parser.add_argument('--Aircraft_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--Omniglot_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Omniglot (default: 4).')
    parser.add_argument('--Quickdraw_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Quickdraw (default: 4).')
    parser.add_argument('--Logo_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Logo (default: 4).')



    parser.add_argument('--num-batches', type=int, default=200,
        help='Number of batches the prototypical network is trained over (default: 200).')
    parser.add_argument('--num_valid_batches', type=int, default=150,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--sample', type=int, default=1,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--memory_limit', type=int, default=200,
        help='memory buffer size(default: 200).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--num_query', type=int, default=10,
        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')
    parser.add_argument('--Reservior', action='store_true',
        help='Use Reservoir Sampling or not.')
    parser.add_argument('--memory_only', action='store_true',
        help='Use memory only.')
    parser.add_argument('--num_epoch', type=int, default=40,
        help='Number of epochs for meta train.')
    parser.add_argument("--epochs", nargs="+", default=[25, 10, 30, 10, 10, 120])
    parser.add_argument('--valid_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for validation (default: 4).')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('args.device', args.device)
    main(args)

