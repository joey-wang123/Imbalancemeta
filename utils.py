import torch
import random
import torch.nn.functional as F
import os
import numpy as np
from scipy.spatial.distance import cdist


def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)

    prototypes -= prototypes.min(-1, keepdim=True)[0]
    prototypes /= prototypes.max(-1, keepdim=True)[0]

    embeddings -= embeddings.min(-1, keepdim=True)[0]
    embeddings /= embeddings.max(-1, keepdim=True)[0]

    norm_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)

    tau = 1.0
    norm_distances = norm_distances/tau
    softprob = -1.0*F.softmax(norm_distances, dim=-1) * F.log_softmax(norm_distances, dim=-1)
    min_dist, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float()), softprob, predictions




def gauss_kernel(X, y, sigma):
    """
    Gaussian kernel.

    Parameters
    ----------
    X: np.ndarray  (n,d),
    y: np.ndarray  (d,),
    sigma: float,
        variance of the kernel.

    Returns
    -------
        k (n,): kernel between each row of X and y
    """
    return np.squeeze(np.exp(-cdist(X, y[np.newaxis, :], metric='sqeuclidean') / (2 * sigma ** 2)))


def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''

    #print('apply grad.keys()', grad.keys())
    grad_norm = 0
    for name, p in model.named_parameters():
        #if p.requires_grad and 'rho' not in name:
        if p.requires_grad:
            if p.grad is None:
                p.grad = grad[name]
            else:
                p.grad += grad[name]
            grad_norm += torch.sum(grad[name]**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()

def apply_grad_ind(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()


def mix_grad_ind(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad

def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''

    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    '''

    mixed_grad = {}
    index = 0
    #print('g_dict', grad_list[0])
    for g_dict in grad_list:
        for name, grad in g_dict.items():
            if index == 0:
                mixed_grad[name] = grad*weight_list[index]
            else:
                mixed_grad[name] += grad*weight_list[index]
            #g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
            #mixed_grad.append(torch.sum(g_list, dim=0))
        index += 1
    return mixed_grad

def grad_to_cos(grad_list):
    '''
    generate cosine similarity from list of gradient
    '''
    cos = 0.
    for g_list in zip(*grad_list):
        g_list = torch.stack(g_list)
        g_list = g_list.reshape(g_list.shape[0], -1) # (n, p)
        g_sum = torch.sum(g_list,dim=0) # (p)
        cos += torch.sum(g_list * g_sum.unsqueeze(0), dim=1) # (n)
    cos = cos/torch.sum(cos)
    return cos


def get_accuracy_ANIL(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def loss_to_ent(loss_list, lamb=1.0, beta=1.0):
    '''
    generate entropy weight from list of loss (uncertainty in loss function)
    '''
    loss_list = np.array(loss_list)
    ent = 1./(lamb + beta * loss_list)
    return ent

def set_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def set_gpu(x):
    x = [str(e) for e in x]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(x)
    print('using gpu:', ','.join(x))

def check_dir(args):
    # save path
    path = os.path.join(args.result_path, args.alg)
    if not os.path.exists(path):
        os.makedirs(path)
    return None