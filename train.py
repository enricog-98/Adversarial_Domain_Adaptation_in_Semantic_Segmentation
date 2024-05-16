#Training and validation loops
import torch
import time
from tqdm import tqdm
from utils import poly_lr_scheduler

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, device, n_epochs, lr_schedule, model_name):
    best_iou = 0.0
    best_epoch = 0
    
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        train_intersection = 0
        train_union = 0
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        for i, (inputs, labels) in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if lr_schedule is True:
                poly_lr_scheduler(optimizer, init_lr=2.5e-2, iter=epoch, max_iter=n_epochs)       
            optimizer.step()
            
            predictions = torch.argmax(outputs, dim=1)
            train_intersection += torch.logical_and(labels == predictions, labels != 255).sum()
            train_union += torch.logical_or(labels == predictions, labels != 255).sum()

            train_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Train)')
            
        train_iou = 100*train_intersection/train_union
        
        model.eval()
        test_intersection = 0
        test_union = 0
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False)
        with torch.no_grad():
            for i, (inputs, labels) in test_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                predictions = torch.argmax(outputs, dim=1)                
                test_intersection += torch.logical_and(labels == predictions, labels != 255).sum()
                test_union += torch.logical_or(labels == predictions, labels != 255).sum()

                test_loop.set_description(f'Epoch {epoch+1}/{n_epochs} (Test)')
                            
        test_iou = 100*test_intersection/test_union

        # Save a checkpoint after each epoch
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_iou': train_iou,
            'test_iou': test_iou,
            'best_iou': best_iou,
            'best_epoch': best_epoch
        }
        torch.save(checkpoint, f'checkpoints/{model_name}_checkpoint_epoch_{epoch+1}.pth')
        
        #Early stopping condition
        if test_iou > best_iou:
            best_iou = test_iou
            best_epoch = epoch
            torch.save(checkpoint, f'checkpoints/{model_name}_best_epoch_{epoch+1}.pth')

        end = time.time()

        print(f'Epoch {epoch+1}/{n_epochs}, Train IoU: {train_iou:.2f}% ({train_intersection}/{train_union}), Test IoU: {test_iou:.2f}% ({test_intersection}/{test_union}) [{(end-start) // 60:.0f}m {(end-start) % 60:.0f}s]')

    print(f'Best IoU: {best_iou:.2f}% at epoch {best_epoch+1}')        