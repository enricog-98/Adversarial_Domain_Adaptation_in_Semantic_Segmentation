#Training and validation loops
import torch
import time
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from utils import poly_lr_scheduler, fast_hist, per_class_iou

def train_adversarial(gen, dis, g_criterion, d_criterion, g_optimizer, d_optimizer, lambda_adv, s_dataloader, t_dataloader, class_names, device, n_epochs, model_name):    
    n_classes = len(class_names)
    best_miou = 0.0
    best_class_iou = np.zeros(n_classes)
    best_epoch = 0
    all_train_miou = []
    all_test_miou = []

    #Initialize the labels for the adversarial training
    source_label = 0
    target_label = 1
    
    for epoch in range(n_epochs):

        start = time.time()      

        gen.train()
        dis.train()

        #train_hist = np.zeros((n_classes, n_classes))

        #Train G
        #Don't accumulate gradients in D
        for param in dis.parameters():
                param.requires_grad = False

        source_train_loop = tqdm(enumerate(s_dataloader), total=len(s_dataloader), leave=False)
        #Train G with source data
        for i, (inputs, labels) in source_train_loop:
            if i == 10:
                break
            inputs, labels = inputs.to(device), labels.to(device)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            pred1, _, _ = gen(inputs)
            seg_loss = g_criterion(pred1, labels)
            seg_loss.backward()

        target_train_loop = tqdm(enumerate(t_dataloader), total=len(t_dataloader), leave=False)
        #Train G with target data
        for i, (inputs, _) in target_train_loop:
            if i == 3:
                break
            inputs = inputs.to(device)

            pred_target1, _, _ = gen(inputs)
            d_out1 = dis(F.softmax(pred_target1, dim=1))#F.softmax(pred_target1, dim=1))
            adv_loss = d_criterion(d_out1, source_label)
            d_loss = lambda_adv * adv_loss
            d_loss.backward()

            #Train D

            #Bring back gradients in D
            for param in dis.parameters():
                param.requires_grad = True

            #Train D with source data
            d_out1 = dis(pred1.detach())#F.softmax(pred1, dim=1)
            d_loss = d_criterion(d_out1, source_label)
            d_loss.backward()

            #Train D with target data
            d_out1 = dis(pred_target1.detach())#F.softmax(pred_target1, dim=1)
            d_loss = d_criterion(d_out1, target_label)
            d_loss.backward()


            
            g_optimizer.step()
            d_optimizer.step()

            predictions = torch.argmax(outputs, dim=1)

            #train_hist += fast_hist(labels.cpu().numpy(), predictions.cpu().numpy(), n_classes)
            #train_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Train)')
            
        #train_class_iou = 100*per_class_iou(train_hist)
        #train_miou = np.mean(train_class_iou)
        #all_train_miou.append(train_miou)
        
        gen.eval()
        test_hist = np.zeros((n_classes, n_classes))
        test_loop = tqdm(enumerate(t_dataloader), total=len(t_dataloader), leave=False)
        with torch.no_grad():
            for i, (inputs, labels) in test_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = gen(inputs)
                
                predictions = torch.argmax(outputs, dim=1)
                test_hist += fast_hist(labels.cpu().numpy(), predictions.cpu().numpy(), n_classes)
                test_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Test)')
                            
        test_class_iou = 100*per_class_iou(test_hist)
        test_miou = np.mean(test_class_iou)
        all_test_miou.append(test_miou)
        
        #Create a checkpoint dictionary
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': gen.state_dict(),
            'optimizer_state_dict': g_optimizer.state_dict(),
            #'train_class_iou': train_class_iou,
            #'train_miou': train_miou,
            'test_class_iou': test_class_iou,
            'test_miou': test_miou,
        }

        #torch.save(checkpoint, f'checkpoints/{model_name}_checkpoint_epoch_{epoch+1}.pth')
        
        #Early stopping condition
        if test_miou > best_miou:
            best_miou = test_miou
            best_class_iou = test_class_iou
            best_epoch = epoch
            #torch.save(checkpoint, f'checkpoints/{model_name}_best_epoch_{epoch+1}.pth')

        end = time.time()

        #print(f'\nEpoch {epoch+1}/{n_epochs} [{(end-start) // 60:.0f}m {(end-start) % 60:.0f}s]: Train mIoU={train_miou:.2f}%, Test mIoU={test_miou:.2f}%')
        print(f'\nEpoch {epoch+1}/{n_epochs} [{(end-start) // 60:.0f}m {(end-start) % 60:.0f}s]: Test mIoU={test_miou:.2f}%')
        for class_name, iou in zip(class_names, test_class_iou):
            print(f'{class_name}: {iou:.2f}%', end=' ')

    print(f'\nBest mIoU={best_miou:.2f}% at epoch {best_epoch+1}')
    for class_name, iou in zip(class_names, best_class_iou):
        print(f'{class_name}: {iou:.2f}%', end=' ')

    return all_train_miou, all_test_miou, best_epoch