import numpy as np
import torch
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
import matplotlib.pyplot as plt


def plot_miou_over_epochs(all_train_miou, all_test_miou, best_epoch, model_name):
    plt.plot(all_train_miou, label='Train')
    plt.plot(all_test_miou, label='Test')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU%')
    plt.legend()
    plt.savefig(f'plots/{model_name}_mIoU_over_epochs.png')
    plt.show()


def test_latency_FPS(model, device, height, width):
    image = torch.randn(8, 3, height, width).to(device)
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
    image = torch.zeros(8, 3, height, width).to(device)
    flops = FlopCountAnalysis(model, image)
    return flop_count_table(flops)


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, lr_decay_iter=1, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    #if iter % lr_decay_iter or iter > max_iter:
        #return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr


def fast_hist(label, predict, num_classes):
    k = (label >= 0) & (label < num_classes)
    return np.bincount(num_classes * label[k].astype(int) + predict[k], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)