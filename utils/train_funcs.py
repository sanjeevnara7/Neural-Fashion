# Will contain utility functions used for training the model(s)
import torch
import os
import copy
from tqdm import tqdm
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import time
from utils.BLEU import compute_bleu

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
def fit(model, train_loader, val_loader, vocab, optimizer, criterion, epochs=5, device='cpu', name='decoder'):
    '''
    fit() function to train captioning model.

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
    train_bleu_history = []
    val_loss_history = []
    val_bleu_history = []
    best_bleu = 0.
    
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
            running_bleu_scores = torch.tensor([0.]).to(device, non_blocking=True)
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
                        preds = get_predictions(outputs, shape=(inputs.shape[0],outputs.shape[1]), device=device)
                        #loss = torch.tensor([0.])
                        #preds = torch.zeros((outputs.shape))
                        # Do backprop if phase = train
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    bleu_scores = torch.tensor([0.]).to(device, non_blocking=True)
                    for idx in range(captions.shape[0]):
                        w1, w2 = seq2text(preds[idx], captions[idx], vocab)
                        #print('Pred: ', w1)
                        #print('Caption: ', w2)
                        #print('*'*15)
                        score = compute_bleu([[w2]], [w1])
                        #print('Score: ',score)
                        bleu_scores += score[0]
                    
#                     if itr % 50 == 0:
#                         w1, w2 = seq2text(preds[0], captions[0], vocab)
#                         score = compute_bleu([[w2]], [w1])
#                         desc = '\n Pred: '+str(w1)+'\n Caption: '+str(w2)+' '+str(score)
#                         print(desc)
                        
                    if phase == 'train':
                        writer.add_scalar("Loss/"+phase, loss.item(), epoch * len(data_loader) + itr)
                        writer.add_scalar("BLEU score/"+phase,
                                          (bleu_scores/(captions.shape[0])).item(),
                                          epoch * len(data_loader) + itr)
                    
                    tepoch.set_postfix(loss=loss.item(), BLEU = (bleu_scores/(captions.shape[0])).item())
                    running_bleu_scores += bleu_scores / captions.shape[0]
                
                epoch_loss = running_loss / len(data_loader.dataset)
                epoc_bleu = running_bleu_scores / len(data_loader)
                print(f"Epoch {(epoch+1)} {phase} loss: {epoch_loss} {phase} BLEU: {epoc_bleu.item()}")
                
                if phase == 'val':
                    writer.add_scalar("Loss/"+phase, epoch_loss, epoch+1)
                    writer.add_scalar("BLEU score/"+phase, epoc_bleu, epoch+1)

                if phase == 'val':
                    val_loss_history.append(epoch_loss)
                    val_bleu_history.append(epoc_bleu.item())
                else:
                    train_loss_history.append(epoch_loss)
                    train_bleu_history.append(epoc_bleu.item())
                
                #Saving best model
                if phase == 'val' and epoc_bleu > best_bleu:
                    best_bleu = epoc_bleu
                    os.makedirs('./models', exist_ok = True)
                    torch.save({      
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, f"./models/{name}_model.pth")
                
        print('-'*20)
        epoch_end_time = time.time()
    #End of Training    
    writer.close()
    print('Best BLEU Score: {}'.format(best_bleu.item()))
    print(f"Time taken for an epoch: {epoch_end_time - epoch_start_time}")
    return (train_loss_history, train_bleu_history, val_loss_history, val_bleu_history)


#Loss function for decoder
def decoder_loss(outputs, captions, criterion, vocab):
    '''
    Loss function that calculates cross-entropy over each predicted output word and sums it.

    args:
        outputs - a tensor of outputs where each output corresponds to a vector of predictions
        captions - a tensor of ground truth caption

    '''
    loss_out = torch.tensor([0.]).cuda()
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            loss_out += criterion(outputs[i, j], captions[i, j+1]) #Current word pred = next word in ground truth
            # stop calculating loss when caption reaches eos
            if(captions[i, j+1] == vocab['eos']):
                break 
    return loss_out/outputs.shape[0]

#Utility function to get predictions
def get_predictions(outputs, shape, device):
    '''
    Utility function that returns predictions for a list of outputs

    args:
        outputs - a list of outputs where each output corresponds to a vector of predictions
        shape - shape of the predictions to return
    '''
    preds = torch.empty(size=shape).to(device, non_blocking=True)
    for i in range(outputs.shape[1]):
        preds[:,i] = torch.argmax(outputs[:, i, :], dim=-1)
    return preds
    
#Utility function to convert to sentences
def seq2text(pred, caption, vocab):
    rev_vocab = {v: k for k, v in vocab.items()}
    
    #Truncate to first <eos> token
#     for i in range(pred.shape[0]):
#         if pred[i] == vocab['eos']:
#             pred = pred[:i]
#             break
    for i in range(caption.shape[0]):
        if caption[i] == vocab['eos']:
            caption = caption[:i+1]
            break
    #print(pred, caption, pred.shape, caption.shape)
    #Build Strings
    
    w1, w2 = [], []
    for i in range(pred.shape[0]):
        #print("Pred shape: ", pred.shape)
        word = rev_vocab[pred[i].int().item()]
        w1.append(word)
    
    for i in range(1, caption.shape[0]):
        word = rev_vocab[caption[i].int().item()]
        w2.append(word)
        
    return w1, w2
    




