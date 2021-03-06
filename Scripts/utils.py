import torch
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def idx2onehot(idx, n):

    assert idx.size(1) == 1
    assert torch.max(idx).data[0] < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, torch.LongTensor(idx.data.cpu()), 1)
    onehot = to_var(onehot)
    
    return onehot