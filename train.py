#Training and validation loops
import torch

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, device, n_epochs):
    for epoch in range(n_epochs):
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
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_dataloader)
        
        model.eval()
        test_loss = 0.0
        intersection = 0
        union = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                if i == 3:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                
                intersection += torch.logical_and(labels == predictions, labels != 255).sum()
                union += torch.logical_or(labels == predictions, labels != 255).sum()
                
            test_loss /= len(test_dataloader)
        
        print(f'Epoch {epoch+1}/{n_epochs}, IoU: {100*intersection/union:.2f}%, Intersection: {intersection}, Union: {union}')