import torch
import random
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
import torch.nn.functional as F
import torch.nn as nn
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


