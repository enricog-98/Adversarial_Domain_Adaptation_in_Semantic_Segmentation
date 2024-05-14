#Training and validation loops
import torch
import time
from utils import poly_lr_scheduler

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, device, n_epochs, lr_schedule):
    best_iou = 0.0
    best_epoch = 0
    
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            if i == 10:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()

            if lr_schedule is True:
                poly_lr_scheduler(optimizer, init_lr=2.5e-2, iter=epoch, max_iter=n_epochs)
                        
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_dataloader)
        
        model.eval()
        test_loss = 0.0
        intersection = 0
        union = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                if i == 5:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                
                intersection += torch.logical_and(labels == predictions, labels != 255).sum()
                union += torch.logical_or(labels == predictions, labels != 255).sum()
                
            test_loss /= len(test_dataloader)
            
        iou = 100*intersection/union
        if iou > best_iou:
            best_iou = iou
            best_epoch = epoch

        end = time.time()

        print(f'Epoch {epoch+1}/{n_epochs}, IoU: {100*intersection/union:.2f}% ({intersection}/{union}) [{end-start:.2f}s]')

    print(f'Best IoU: {best_iou:.2f}% at epoch {best_epoch+1}')        