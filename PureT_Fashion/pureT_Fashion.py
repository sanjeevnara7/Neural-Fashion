############################################################################
# Implementation taken from the PureTransformer repository
# https://github.com/232525/PureT 
# @inproceedings{wangyiyu2022PureT,
#   title={End-to-End Transformer Based Model for Image Captioning},
#   author={Yiyu Wang and Jungang Xu and Yingfei Sun},
#   booktitle={AAAI},
#   year={2022}
# }
############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils as utils
from basic_model import BasicModel

from PureT_encoder import Encoder
from PureT_decoder import Decoder

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    return subsequent_mask == 0

class PureT_Fashion(BasicModel):
    def __init__(self):
        super(PureT_Fashion, self).__init__()
        
        self.vocab_size =  109

        # Useful in later functions
        self.att_feats = None
        self.att_mask = None
        self.gv_feat = None

        """Feature map from swin backbone in the PureT -- (12 * 12) x 1536"""
        #Attention Feature Dimension -- 1536
        #Embedding dimension of PureT -- 512

        """Our implementation (8 * 8) x 768"""
        #Attention Feature Dimension -- 768

        self.att_embed = nn.Sequential(
                nn.Linear(768, 512),
                utils.activation('CELU'),
                nn.Identity(),
                nn.Dropout(0.1)
            )
        
        # This helps use a global vector to add to input feature map
        use_gx = True

        self.encoder = Encoder(
            embed_dim=512, 
            input_resolution=(8, 8), 
            depth=2,        #this can be played with, original value = 3 
            num_heads=4,    #this can be played with, original value = 8
            window_size=4,  #input sent is (8 * 8) x 768
            shift_size=3,
            mlp_ratio=4,
            dropout=0.1,
            use_gx = use_gx
        )
        
        self.decoder = Decoder(
            vocab_size = self.vocab_size, 
            embed_dim = 512, 
            depth = 2,        #this can be played with, original value = 3
            num_heads = 4,    #this can be played with, original value = 8
            dropout = 0.1, 
            ff_dropout = 0.1,
            use_gx = use_gx
        )

    def forward(self, att_feats, seq, gv_feat):#, **kwargs):
        
        """Dimensions of the inputs"""
        # seq is of size B x 109
        # att_feats are directly sent from the swin transformer backbone we trained
        # att_feats are of size B x (8 * 8) x 768
        # gv_feat is just a B x [[0]] tensor initially

        att_mask = torch.ones(att_feats.size()[0], 8 * 8).cuda()
        self.att_feats = att_feats
        self.att_mask = att_mask
        self.gv_feat = gv_feat

        # words mask [B, L, L]
        ##############################################
        """Ignore padded input during the masking"""
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:,0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)

        decoder_out = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)
        return F.log_softmax(decoder_out, dim=-1)

    def get_logprobs_state(self):
        wt = self.param_wt             
        state = self.param_state        
        encoder_out = self.att_feats    
        
        att_mask = self.att_mask        
        gx = self.gv_feat               

        # state[0][0]: [B, seq_len-1]，previously generated words
        # ys: [B, seq_len]
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
            
        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(1)
        
        # [B, 1, Vocab_Size] --> [B, Vocab_Size]
        decoder_out = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, seq_mask, att_mask).squeeze(1)
        
        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn

    def decode(self, att_feats):
        greedy_decode = True
        att_feats = self.att_feats
        att_mask = self.att_mask

        batch_size = att_feats.size(0)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        self.decoder.init_buffer(batch_size)
        
        state = None
        sents = Variable(torch.zeros((batch_size, self.MODEL_SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, self.MODEL_SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        self.att_feats = encoder_out
        self.gv_feat = gx
        
        # inference word by word
        for t in range(self.MODEL_SEQ_LEN):
            self.param_wt = wt
            self.param_state = state
            logprobs_t, state = self.get_logprobs_state()
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs