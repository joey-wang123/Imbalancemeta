import torch
from tqdm import tqdm
from gbml.maml_Reservoir import MAML
from utils import set_seed, set_gpu, check_dir, gauss_kernel
from datasets import *
import random
import pickle
import domain_shift as algos
import warnings
warnings.filterwarnings("ignore")

datanames = ['Quickdraw', 'MiniImagenet', 'Omniglot', 'CUB', 'Aircraft', 'Necessities']

class Imbalance_ANIL(object):

    def __init__(self,model, args=None):
        self.args = args
        self.model=model
        self.step = 0
        self.domain_id  = 0
        self.memory_rep = []
        self.window = []
        self.domain_iter = {}
        self.domain_iter['0'] = 0
        for ind in range(50):
            self.domain_iter[str(ind)] = 0

        self.domain_embed = 0.0
        self.numsteps = 5
        self.startiter = 200
        self.detect_shift = True

        B = 200 #window size
        big_Lambda, small_lambda = algos.select_optimal_parameters(B)  # forget factors chosen with heuristic in the paper
        thres_ff = small_lambda
        sigmasq = 150.0
        X0 = np.array([60.58, 65.70, 61.91, 69.66, 71.77])
        N = 3
        self.detector = algos.Online_domainshift_detection(X0, kernel_func=lambda x, y:gauss_kernel(x, y, np.sqrt(sigmasq)), window_size=B,
                               nbr_windows=N, adapt_forget_factor=thres_ff)

        str_save = '_'.join(datanames)
        self.filepath = os.path.join(self.args.output_folder, 'MAML_Reservoirtest_{}_ScanB_{}'.format(str_save, self.args.memory_limit), 'shot{}'.format(args.train_shot), 'way{}'.format(args.num_way))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

    def train(self, epoch, dataloader_dict, domain_id):
        for dataname, dataloader in dataloader_dict.items():
            with tqdm(dataloader, total=self.args.num_train_batches) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    batch = [batch]
                    loss_log, acc_log, grad_log, embed_mean = self.model.outer_loop(batch, is_train=True, memory_train = self.memory_rep)

                    if self.detect_shift:
                        alpha = 0.5
                        self.domain_embed = alpha*embed_mean + (1-alpha)*self.domain_embed
                        lenwindow = 20
                        if (len(self.window))> lenwindow:
                            self.window.remove(self.window[0])
                        
                        if self.window:
                            if (len(self.window)) == lenwindow:
                                dist_list = []
                                for proto in self.window[-1*(self.numsteps+1):-1]:
                                    currentdist = torch.sum((embed_mean - proto)**2).item()
                                    dist_list.append(currentdist)

                            if self.step>self.startiter:
                                x = torch.tensor(dist_list).cpu().detach().numpy()
                                self.detector.update(x)
                                result = self.detector.flag_sample()
                                if result and self.domain_iter[str(self.domain_id)]>600:
                                    self.domain_id = self.domain_id + 1
                                self.domain_iter[str(self.domain_id)]+= 1
                        self.window.append(self.domain_embed)

                        if self.domain_id >= (len(datanames)-1):
                            self.detect_shift = False

                    #Reservoir sampling
                    if self.step < self.args.memory_limit:
                        savedict = {domain_id: batch[0]}
                        self.memory_rep.append(savedict)

                    else:
                        randind = random.randint(0, self.step)
                        if randind < self.args.memory_limit:
                            savedict = {domain_id: batch[0]}
                            self.memory_rep[randind] = savedict
                    self.step = self.step+1

                    if batch_idx >= self.args.num_train_batches:
                        break

    @torch.no_grad()
    def valid(self, epoch, dataloader_dict):

        acc_list = []
        for dataname, dataloader in dataloader_dict.items():
            with torch.no_grad():
                with tqdm(dataloader, total=self.args.num_valid_batches) as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        batch = [batch]
                        loss_log, acc_log = self.model.outer_loop(batch, is_train=False)
                        acc_list.append(acc_log[0])
                        pbar.set_description('dataname {} accuracy ={:.4f}'.format(dataname, np.mean(acc_list)))
                        if batch_idx >= self.args.num_valid_batches:
                            break
        avg_accuracy = np.round(np.mean(acc_list), 4)
        acc_dict = {dataname:avg_accuracy}

        return acc_dict


    def save(self, epoch):
        filename = os.path.join(self.filepath, 'epoch{0}.pt'.format(epoch))
        with open(filename, 'wb') as f:
            state_dict = self.model.network.state_dict()
            torch.save(state_dict, f)

    def load(self, epoch):
        filename = os.path.join(self.filepath, 'epoch{0}.pt'.format(epoch))
        print('loading model filepath', filename)
        self.model.network.load_state_dict(torch.load(filename))



def main(args):


    model = MAML(args)
    train_loader_list, valid_loader_list, test_loader_list = dataset(args, datanames)
    all_accdict = {}
    appr = Imbalance_ANIL(model, args)
    total_epoch = 0
    for loaderindex, train_loader in enumerate(train_loader_list):
        each_epoch = args.epochs[loaderindex]
        for epoch in range(total_epoch, total_epoch+each_epoch):
            print('Epoch {}'.format(epoch))
            appr.train(epoch, train_loader, loaderindex)
            print('domain_id', appr.domain_id)
            epoch_acc = []
            for index, test_loader in enumerate(test_loader_list[:loaderindex+1]):
                test_accuracy_dict = appr.valid(epoch, test_loader)
                epoch_acc.append(test_accuracy_dict)
            all_accdict[str(epoch)] = epoch_acc
            with open(appr.filepath  + '/stats_acc.pickle', 'wb') as handle:
                pickle.dump(all_accdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        total_epoch += each_epoch

    return None



def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Imbalanced ANIL')
    # experimental settings
    parser.add_argument('--seed', type=int, default=2020,
        help='Random seed.')   
    parser.add_argument('--data_path', type=str, default='meta_data/',
        help='Path of MiniImagenet.')
    parser.add_argument('--sequential', action='store_true',
        help='Use sequential training only.')
    parser.add_argument('--memory_only', action='store_true',
        help='Use memory only.')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_encoder', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers for data loading (default: 4).')
    # training settings
    parser.add_argument('--batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks (default: 4).')

    parser.add_argument('--valid_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for validation (default: 4).')
    parser.add_argument('--num_train_batches', type=int, default=200,
        help='Number of batches the model is trained over (default: 250).')
    parser.add_argument('--num_valid_batches', type=int, default=150,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--sample', type=int, default=1,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--memory_limit', type=int, default=200,
        help='memory buffer size (default: 200).')


    # meta-learning settings
    parser.add_argument('--output_folder', type=str, default='output/newsavedir/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--train-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5). during training')
    parser.add_argument('--test-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5) during testing.')
    parser.add_argument('--num_query', type=int, default=10,
        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--num_way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument("--epochs", nargs="+", default=[25, 10, 30, 10, 10, 20])
    # algorithm settings
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)
    # network settings
    parser.add_argument('--net', type=str, default='ConvNet')
    parser.add_argument('--n_conv', type=int, default=4)
    parser.add_argument('--n_dense', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=48)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=48,
        help='Number of channels for each convolutional layer (default: 64).')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    main(args)