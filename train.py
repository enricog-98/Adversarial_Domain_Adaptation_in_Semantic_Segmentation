#Training and validation loops
import torch
import time
import numpy as np
from tqdm import tqdm
from utils import poly_lr_scheduler, fast_hist, per_class_iou

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, class_names, device, n_epochs, model_name):    
    n_classes = len(class_names)
    #initial_lr = optimizer.param_groups[0]['lr']
    best_miou = 0.0
    best_class_iou = np.zeros(n_classes)
    best_epoch = 0
    all_train_miou = []
    all_test_miou = []
    
    for epoch in range(n_epochs):
        
        start = time.time()
        
        model.train()
        train_hist = np.zeros((n_classes, n_classes))
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)        
        for i, (inputs, labels) in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)

            loss = criterion(outputs, labels)    
            loss.backward()

            #poly_lr_scheduler(optimizer, init_lr=initial_lr, iter=epoch, max_iter=n_epochs)       
            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)

            train_hist += fast_hist(labels.cpu().numpy(), predictions.cpu().numpy(), n_classes)
            train_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Train)')
            
        train_class_iou = 100*per_class_iou(train_hist)
        train_miou = np.mean(train_class_iou)
        all_train_miou.append(train_miou)
        
        model.eval()
        test_hist = np.zeros((n_classes, n_classes))
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False)
        with torch.no_grad():
            for i, (inputs, labels) in test_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                predictions = torch.argmax(outputs, dim=1)
                test_hist += fast_hist(labels.cpu().numpy(), predictions.cpu().numpy(), n_classes)
                test_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Test)')
                            
        test_class_iou = 100*per_class_iou(test_hist)
        test_miou = np.mean(test_class_iou)
        all_test_miou.append(test_miou)
        
        #Create a checkpoint dictionary
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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