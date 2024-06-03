#Training and validation loops
import torch
import time
import numpy as np
from tqdm import tqdm
from itertools import cycle
from torch.nn import functional as F
from utils import poly_lr_scheduler, fast_hist, per_class_iou

def train_adversarial(g_model, d_model, g_criterion, d_criterion, g_optimizer, d_optimizer, source_flag, target_flag, s_dataloader, t_dataloader, interp_s, interp_t, class_names, device, n_epochs, model_name):
    n_classes = len(class_names)
    lambda_adv = 0.001
    #g_initial_lr = g_optimizer.param_groups[0]['lr']
    #d_initial_lr = d_optimizer.param_groups[0]['lr']
    best_miou = 0.0
    best_class_iou = np.zeros(n_classes)
    best_epoch = 0
    all_train_miou = []
    all_test_miou = []

    for epoch in range(n_epochs):

        start = time.time()

        g_model.train()
        d_model.train()
        train_hist = np.zeros((n_classes, n_classes))
        t_dataloader_cycle = cycle(t_dataloader)
        train_loop = tqdm(zip(s_dataloader, t_dataloader_cycle), total=len(s_dataloader), leave=False)
        for (source_data, source_labels), (target_data, _) in train_loop:
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            #TRAIN G

            #Do not accumulate gradients for D
            for param in d_model.parameters():
                param.requires_grad = False

            #Train with source
            g_source_output, _, _ = g_model(source_data)
            g_source_output = interp_s(g_source_output)
            
            loss_seg = g_criterion(g_source_output, source_labels)
            loss_seg.backward()

            #Train with target
            g_target_output, _, _ = g_model(target_data)
            g_target_output = interp_t(g_target_output)
            
            d_out = d_model(F.softmax(g_target_output))

            loss_adv_t = d_criterion(d_out, torch.full_like(d_out, source_flag))
            loss_d = lambda_adv * loss_adv_t
            loss_d.backward()

            #TRAIN D

            #Bring back gradients for D
            for param in d_model.parameters():
                param.requires_grad = True

            #Train with source
            g_source_output = g_source_output.detach()
            
            d_out = d_model(F.softmax(g_source_output))
            
            loss_d = d_criterion(d_out, torch.full_like(d_out, source_flag))
            loss_d.backward()

            #Train with target
            g_target_output = g_target_output.detach()
            
            d_out = d_model(F.softmax(g_target_output))
            
            loss_d = d_criterion(d_out, torch.full_like(d_out, target_flag))
            loss_d.backward()

            #Update weights
            #poly_lr_scheduler(g_optimizer, init_lr=g_initial_lr, iter=epoch, max_iter=n_epochs)
            g_optimizer.step()
            #poly_lr_scheduler(d_optimizer, init_lr=d_initial_lr, iter=epoch, max_iter=n_epochs)
            d_optimizer.step()
            
            predictions = torch.argmax(g_source_output, dim=1)

            train_hist += fast_hist(source_labels.cpu().numpy(), predictions.cpu().numpy(), n_classes)
            train_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Train)')

        train_class_iou = 100*per_class_iou(train_hist)
        train_miou = np.mean(train_class_iou)
        all_train_miou.append(train_miou)

        g_model.eval()
        test_hist = np.zeros((n_classes, n_classes))
        test_loop = tqdm(enumerate(t_dataloader), total=len(t_dataloader), leave=False)
        with torch.no_grad():
            for i, (target_data, target_labels) in test_loop:
                target_data, target_labels = target_data.to(device), target_labels.to(device)

                g_target_output = g_model(target_data)

                predictions = torch.argmax(g_target_output, dim=1)
                test_hist += fast_hist(target_labels.cpu().numpy(), predictions.cpu().numpy(), n_classes)
                test_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Test)')

        test_class_iou = 100*per_class_iou(test_hist)
        test_miou = np.mean(test_class_iou)
        all_test_miou.append(test_miou)

        #Create a checkpoint dictionary
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': g_model.state_dict(),
            'optimizer_state_dict': g_optimizer.state_dict(),
            'train_class_iou': train_class_iou,
            'train_miou': train_miou,
            'test_class_iou': test_class_iou,
            'test_miou': test_miou,
        }

        torch.save(checkpoint, f'checkpoints/{model_name}_checkpoint_epoch_{epoch+1}.pth')
        
        #Early stopping condition
        if test_miou > best_miou:
            best_miou = test_miou
            best_class_iou = test_class_iou
            best_epoch = epoch
            torch.save(checkpoint, f'checkpoints/{model_name}_best_epoch_{epoch+1}.pth')

        end = time.time()

        print(f'\nEpoch {epoch+1}/{n_epochs} [{(end-start) // 60:.0f}m {(end-start) % 60:.0f}s]: Train mIoU={train_miou:.2f}%, Test mIoU={test_miou:.2f}%')
        for class_name, iou in zip(class_names, test_class_iou):
            print(f'{class_name}: {iou:.2f}%', end=' ')

    print(f'\nBest mIoU={best_miou:.2f}% at epoch {best_epoch+1}')
    for class_name, iou in zip(class_names, best_class_iou):
        print(f'{class_name}: {iou:.2f}%', end=' ')

    return all_train_miou, all_test_miou, best_epoch