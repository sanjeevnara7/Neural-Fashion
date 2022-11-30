# Will contain utility functions used for training the model(s)
import torch
import copy

#Training Function
def fit_classifier(model, train_loader, val_loader, optimizer, criterion, epochs=10, device='cpu'):
    '''
    fit() function to train a classifier model.

    args:
        model - the model to be trained
        train_loader - torch.utils.data.Dataloader() for train set
        val_loader - torch.utils.data.Dataloader() for val set
        optimizer - optimization algorithm for weight updates
        criterion - loss function to be used for training
    
    keyword args:
        epochs - Number of training epochs (default=10)
        device - the device for training (default='cpu')
    
    '''
    if device is not 'cpu':
        model.to(device)
    loss_history = []
    acc_history = []
    best_acc = 0.
    #Iterate epochs
    for epoch in range(epochs):
        print('Training epoch {}/{}...:'.format(epoch+1, epochs))
        #Each epoch has a training phase and validation phase
        for phase in ['train','val']:
            data_loader = None
            if phase == 'train':
                #Set train mode
                model.train()
                data_loader = train_loader
            else:
                #Set Eval mode
                model.eval()
                data_loader = val_loader

            running_loss = 0.
            running_corrects = 0
            #Iterate batches
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad()
                #Set gradient calc on only for training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = classifier_preds(outputs, shape=(inputs.shape[0],labels.shape[1])).float()
                    #Do backprop if phase = train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.float() / len(data_loader.dataset)
            print('{} loss: {}, {} acc: {}'.format(phase, epoch_loss, phase, epoch_acc))
            if phase == 'val':
                loss_history.append(epoch_loss)
                acc_history.append(epoch_acc)
            #Saving best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print('-'*20)
    print('Best val acc: {}'.format(best_acc))

#Loss function for classifier
def classifier_loss(outputs, targets):
    '''
    Loss function that calculates cross-entropy over each output and sums it.

    args:
        outputs - a list of outputs where each output corresponds to a vector of predictions
        targets - a tensor of targets where each target corresponds to a class index

    '''
    loss_out = []
    for index, output in enumerate(outputs):
        loss_out.append(torch.nn.CrossEntropyLoss(output, targets[:,index]))
    return torch.tensor(loss_out)

#Utility method to get predictions
def classifier_preds(outputs, shape):
    '''
    Utility function that returns predictions for a list of outputs

    args:
        outputs - a list of outputs where each output corresponds to a vector of predictions
        shape - shape of the predictions to return
    '''
    preds = torch.empty(size=shape)
    for index, output in enumerate(outputs):
        preds[:,index] = torch.argmax(output, dim=1)
    return preds

