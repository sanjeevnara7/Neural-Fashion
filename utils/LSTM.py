import torch 
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size = 512, hidden_size = 512, vocab_size = 107, num_layers=3):
        super(DecoderLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.embed_size= embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers, batch_first=True, dropout = 0.2)
        
        self.linear1 = nn.Linear(11*7*512, 95*512)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        
        
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
    # Forward function
    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        
        features = features.reshape(-1,512) # Original shape (11x7x512)
        features = self.linear1(features)
        
        # Concatenate features and embeddings along the columns  
        embeddings = torch.cat((features, embeddings[:, :-1, :]), dim = -1)
        
        hidden_out, _ = self.lstm(embeddings)
        outputs = self.out(hidden_out)
        
        return outputs