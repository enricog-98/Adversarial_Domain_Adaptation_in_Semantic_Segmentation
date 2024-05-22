#Training and validation loops
import torch
import time
import numpy as np
from tqdm import tqdm
from utils import poly_lr_scheduler, fast_hist, per_class_iou

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, class_names, device, n_epochs, lr_schedule, model_name):    
    n_classes = len(class_names)
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
            if i == 15:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if model_name == 'DeepLabV2':
                outputs, _, _ = model(inputs)

                loss = criterion(outputs, labels)

            elif model_name == 'BiSeNet':
                outputs, aux_outputs1, aux_outputs2 = model(inputs)
                #outputs = (outputs + aux_outputs1 + aux_outputs2) / 3                
                
                main_loss = criterion(outputs, labels)
                aux_loss1 = criterion(aux_outputs1, labels)
                aux_loss2 = criterion(aux_outputs2, labels)
                loss = main_loss + 1 * (aux_loss1 + aux_loss2)

            loss.backward()

            if lr_schedule is True:
                poly_lr_scheduler(optimizer, init_lr=2.5e-2, iter=epoch, max_iter=n_epochs)       
            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)

            train_hist += fast_hist(labels.numpy(), predictions.numpy(), n_classes)
            train_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Train)')
            
        train_class_iou = 100*per_class_iou(train_hist)
        train_miou = np.mean(train_class_iou)
        all_train_miou.append(train_miou)
        
        model.eval()
        test_hist = np.zeros((n_classes, n_classes))
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False)
        with torch.no_grad():
            for i, (inputs, labels) in test_loop:
                if i == 5:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                predictions = torch.argmax(outputs, dim=1)
                test_hist += fast_hist(labels.numpy(), predictions.numpy(), n_classes)
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

        #torch.save(checkpoint, f'checkpoints/{model_name}_checkpoint_epoch_{epoch+1}.pth')
        
        #Early stopping condition
        if test_miou > best_miou:
            best_miou = test_miou
            best_class_iou = test_class_iou
            best_epoch = epoch
            #torch.save(checkpoint, f'checkpoints/{model_name}_best_epoch_{epoch+1}.pth')

        end = time.time()

        print(f'Epoch {epoch+1}/{n_epochs} [{(end-start) // 60:.0f}m {(end-start) % 60:.0f}s]')
        print(f'Train mIoU: {train_miou:.2f}%, Test mIoU: {test_miou:.2f}%')
        for class_name, iou in zip(class_names, test_class_iou):
            print(f'{class_name}: {iou:.2f}%', end=' ')

    print(f'\nBest mIoU: {best_miou:.2f}% at epoch {best_epoch+1}')
    for class_name, iou in zip(class_names, best_class_iou):
        print(f'{class_name}: {iou:.2f}%', end=' ')

    return all_train_miou, all_test_miou, best_epoch