# Will contain utility functions used for training the model(s)
import torch
import os
import copy
from tqdm import tqdm
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import time
from nltk.translate.bleu_score import sentence_bleu

#Training Function
def fit_classifier(model, train_loader, val_loader, optimizer, loss_func, attributes, epochs=10, initial_epoch=0, device='cpu', name='transformer'):
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
    
    returns: (train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    
    '''
    
    model = model.to(device, non_blocking=True)
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_acc = 0.
    
    #create the logger object
    writer = SummaryWriter()
    
    #Iterate epochs
    for epoch in range(initial_epoch, initial_epoch+epochs):
        epoch_start_time = time.time()
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
            running_corrects = torch.tensor([0.]).to(device, non_blocking=True)
            with tqdm(data_loader, unit="batch") as tepoch:
                #Iterate batches
                for itr, (inputs, labels, _) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {(epoch+1)} {phase}")
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.long().to(device, non_blocking=True)
                    optimizer.zero_grad()
                    
                    #Set gradient calc on only for training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = classifier_loss(outputs, labels, loss_func, attributes)
                        preds = classifier_preds(outputs, shape=(inputs.shape[0],labels.shape[1]), device=device)
                        #Do backprop if phase = train
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)
                    
                    if phase == 'train':
                        writer.add_scalar("Loss/"+phase, loss.item(), epoch * len(data_loader) + itr)
                        writer.add_scalar("Accuracy/"+phase,
                                          (torch.sum(preds == labels)/(inputs.shape[0] * labels.shape[1])).item(),
                                          epoch * len(data_loader) + itr)
                    
                    tepoch.set_postfix(loss=loss.item(),
                                       accuracy=(torch.sum(preds == labels)/(inputs.shape[0] * labels.shape[1])).item())
                
                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_corrects.float() / (len(data_loader.dataset) * labels.shape[1])
                print(f"Epoch {(epoch+1)} {phase} loss: {epoch_loss} {phase} accuracy: {epoch_acc.item()}")
                
                if phase == 'val':
                    writer.add_scalar("Loss/"+phase, epoch_loss, epoch+1)
                    writer.add_scalar("Accuracy/"+phase, epoch_acc, epoch+1)

                if phase == 'val':
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())
                else:
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                
                #Saving best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    os.makedirs('./models', exist_ok = True)
                    torch.save({      
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, f"./models/{name}_attribute_model.pth")
                    #best_model_wts = copy.deepcopy(model.state_dict())
                
        print('-'*20)
        epoch_end_time = time.time()
    #End of Training    
    writer.close()
    print('Best val acc: {}'.format(best_acc.item()))
    print(f"Time taken for an epoch: {epoch_end_time - epoch_start_time}")
    return (train_loss_history, train_acc_history, val_loss_history, val_acc_history)

#Loss function for classifier
def classifier_loss(outputs, targets, loss_func, attributes):
    '''
    Loss function that calculates cross-entropy over each output and sums it.

    args:
        outputs - a list of outputs where each output corresponds to a vector of predictions
        targets - a tensor of targets where each target corresponds to a class index

    '''
    loss_out = torch.empty((len(attributes), 1))
    for index, output in enumerate(outputs):
        loss_out[index] = loss_func(output, targets[:,index])
    return torch.sum(loss_out)

#Utility method to get predictions
def classifier_preds(outputs, shape, device):
    '''
    Utility function that returns predictions for a list of outputs

    args:
        outputs - a list of outputs where each output corresponds to a vector of predictions
        shape - shape of the predictions to return
    '''
    preds = torch.empty(size=shape).to(device, non_blocking=True)
    for index, output in enumerate(outputs):
        preds[:,index] = torch.argmax(output, dim=1)
    return preds




#Training Function for decoder
def train(model, train_loader, val_loader, vocab, optimizer, criterion, epochs=5, device='cpu'):
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
    
    returns: (train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    
    '''
    
    model = model.to(device, non_blocking=True)
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_acc = 0.
    
    # Create the logger object
    writer = SummaryWriter()
    
    # Iterate epochs
    for epoch in range(epochs):
        epoch_start_time = time.time()
        # Each epoch has a training phase and validation phase
        for phase in ['train','val']:
            data_loader = None
            if phase == 'train':
                #Set train mode
                model.train()
                data_loader = train_loader
            else:
                # Set Eval mode
                model.eval()
                data_loader = val_loader

            running_loss = 0.
            runnning_bleu_scores = torch.tensor([0.]).to(device, non_blocking=True)
            with tqdm(data_loader, unit="batch") as tepoch:
                # Iterate batches
                for itr, (inputs, labels, captions) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {(epoch+1)} {phase}")
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.long().to(device, non_blocking=True)
                    captions = captions.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    
                    # Set gradient calc on only for training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, captions, phase)
                        loss = decoder_loss(outputs, captions, criterion, vocab)
                        preds = get_predictions(outputs, shape=(inputs.shape[0],captions.shape[2]), device=device)
                        # Do backprop if phase = train
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    bleu_scores = torch.tensor([0.]).to(device, non_blocking=True)
                    for idx in range(captions.shape(0)):
                        bleu_scores += sentence_bleu(preds[idx], captions[idx])
                       
                    if phase == 'train':
                        writer.add_scalar("Loss/"+phase, loss.item(), epoch * len(data_loader) + itr)
                        writer.add_scalar("BLEU score/"+phase,
                                          (bleu_scores/(captions.shape[0])).item(),
                                          epoch * len(data_loader) + itr)
                    
                    tepoch.set_postfix(loss=loss.item(), BLEU = (bleu_scores/(captions.shape[0])).item())
                    running_bleu_scores += bleu_scores / captions.shape[0]
                
                epoch_loss = running_loss / len(data_loader.dataset)
                epoc_bleu = running_bleu_scores / len(data_loader)
                print(f"Epoch {(epoch+1)} {phase} loss: {epoch_loss} {phase} accuracy: {epoch_acc.item()}")
                
                if phase == 'val':
                    writer.add_scalar("Loss/"+phase, epoch_loss, epoch+1)
                    writer.add_scalar("BLEU score/"+phase, epoc_bleu, epoch+1)

                if phase == 'val':
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoc_bleu.item())
                else:
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoc_bleu.item())
                
                #Saving best model
                if phase == 'val' and epoc_bleu > best_bleu:
                    best_acc = epoch_acc
                    os.makedirs('./models', exist_ok = True)
                    torch.save({      
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, f"./models/{name}_attribute_model.pth")
                    #best_model_wts = copy.deepcopy(model.state_dict())
                
        print('-'*20)
        epoch_end_time = time.time()
    #End of Training    
    writer.close()
    print('Best val acc: {}'.format(best_acc.item()))
    print(f"Time taken for an epoch: {epoch_end_time - epoch_start_time}")
    return (train_loss_history, train_acc_history, val_loss_history, val_acc_history)


#Loss function for classifier
def decoder_loss(outputs, captions, criterion, vocab):
    '''
    Loss function that calculates cross-entropy over each predicted output word and sums it.

    args:
        outputs - a list of outputs where each output corresponds to a vector of predictions
        captions - a tensor of ground truth caption

    '''
    loss_out = torch.empty((outputs.shape(1)))
    for idx in range(outputs.shape(1)):
        loss_out[idx] = criterion(output[idx], caption[idx])
        
        # stop calculating loss when caption reaches eos
        if(caption[idx] == vocab['<eos>']):
            break 
    return torch.sum(loss_out)

# Utility function to get predictions
def get_predictions(outputs, shape, device):
    '''
    Utility function that returns predictions for a list of outputs

    args:
        outputs - a list of outputs where each output corresponds to a vector of predictions
        shape - shape of the predictions to return
    '''
    preds = torch.empty(size=shape).to(device, non_blocking=True)
    for index, output in enumerate(outputs):
        preds[:,index] = torch.argmax(output, dim=1)
    return preds





