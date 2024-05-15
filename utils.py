import numpy as np
import torch
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table

def test_latency_FPS(model, device, height, width):
    image = torch.randn(1, 3, height, width).to(device)
    iterations = 1000
    latency = []
    FPS = []

    for i in range(iterations):
        start = time.time()
        output = model(image)
        end = time.time()

        latencyi = end - start

        if latencyi != 0.0:
            latency.append(latencyi)
            FPSi = 1/latencyi
            FPS.append(FPSi)

    mean_latency = np.mean(latency)
    std_latency = np.std(latency)
    mean_FPS = np.mean(FPS)
    std_FPS = np.std(FPS)
    
    return 'Mean latency: {:.4f} +/- {:.4f} seconds \nMean FPS: {:.2f} +/- {:.2f} frames per second'.format(mean_latency, std_latency, mean_FPS, std_FPS)

def test_FLOPs_params(model, device, height, width):
    image = torch.zeros(1, 3, height, width).to(device)
    flops = FlopCountAnalysis(model, image)
    return flop_count_table(flops)


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr


def fast_hist(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)