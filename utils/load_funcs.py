# Will contain utility functions to load data
import re
import numpy as np
from torch.utils.data import DataLoader
from utils.customDataset import FashionDataset
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Function to get dataloaders for train/val sets
def get_data_loaders():
    #Read captions
    captions = pd.read_json('./labels/captions.json', typ='series').to_frame()
    data_iter = data_yield(captions)
    #Fit tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>', filters='!"#$%&()*+-/:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data_iter)
    sequences = tokenizer.texts_to_sequences(captions[0].to_list())
    sequences = pad_sequences(sequences, padding='post')
    captions['sequence'] = sequences.tolist()

    #Read Numpy data
    train_data = np.load('./labels/train_data.npy', allow_pickle=True)
    val_data = np.load('./labels/validation_data.npy', allow_pickle=True)
    
    train_dataset = FashionDataset(
        data_np=train_data,
        root_dir='../data/images_224x329',
        mode='train', 
        captions=captions,
        tokenizer=tokenizer
    )
    val_dataset = FashionDataset(
        data_np=val_data,
        root_dir='../data/images_224x329',
        mode='val',
        captions=captions,
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False)
    
    return train_loader, val_loader

#Function to preprocess sentences
def preprocess_sentence(w):
    w = w.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<sos> ' + w + ' <eos>'
    return w

#Generator for sentences
def data_yield(data):
    for index,row in data.iterrows():
        yield preprocess_sentence(row[0])