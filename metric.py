from include import *
import torch
#IMPORT
#PI  = np.pi
#INF = np.inf
#EPS = 1e-12

def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def dice_accuracy(prob, truth, threshold=0.5, is_average=True):

    batch_size = prob.size(0)
    #batch_size = prob.shape[0]
    p = prob.detach().view(batch_size, -1)
    t = truth.detach().view(batch_size, -1)
    #print(p.shape)
    p = p > threshold
    t = t > 0.1
    intersection = p & t
    union = p | t
    #dice = (intersection.float().sum(1) + EPS) / (union.float().sum(1) + EPS)
    dice = (2 * intersection.float().sum(1)+EPS) / (p.float().sum(1) + t.float().sum(1) + EPS)

    if is_average:
        dice = dice.sum() / batch_size
        return dice
    else:
        return dice

def Jaccard_accuracy(prob, truth, threshold=0.5, is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size, -1)
    t = truth.detach().view(batch_size, -1)

    p = p > threshold
    t = t > 0.1
    intersection = p & t
    union = p | t
 #   dice = (intersection.float().sum(1) + EPS) / (union.float().sum(1) + EPS)
    Jac = ( intersection.float().sum(1)+EPS) / (p.float().sum(1) + t.float().sum(1) - intersection.float().sum(1) + EPS)

    if is_average:
        Jac = Jac.sum() / batch_size
        return Jac
    else:
        return Jac


def accuracy(prob, truth, threshold=0.5, is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size, -1)
    t = truth.detach().view(batch_size, -1)

    p = p > threshold
    t = t > 0.5
    correct = (p == t).float()
    accuracy = correct.sum(1) / p.size(1)

    if is_average:
        accuracy = accuracy.sum() / batch_size
        return accuracy
    else:
        return accuracy


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    print('\nsucess!')