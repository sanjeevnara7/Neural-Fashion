# TODO: Add license of the Pure Transformer code

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from lib.config import cfg

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
        
        # need to check this once
        self.vocab_size =  109 + 1 #cfg.MODEL.VOCAB_SIZE + 1

        # Useful in later functions
        self.att_feats = None
        self.att_mask = None
        self.gv_feat = None

        """Feature map from swin backbone in the PureT -- 1536 x 12 x 12"""
        #Attention Feature Dimension -- 1536
        #Embedding dimension of PureT -- 512

        """In our case 768 x 12 x 8"""
        #Attention Feature Dimension -- 768

        """PureT code raw Dimension to Model Dimension"""
        # if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
        #     self.att_embed = nn.Identity()
        # else:
        #     self.att_embed = nn.Sequential(
        #         nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
        #         utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
        #         nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
        #         nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)
        #     )

        # self.encoder = Encoder(
        #     embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM, 
        #     input_resolution=(12, 12), 
        #     depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS, 
        #     num_heads=cfg.MODEL.BILINEAR.HEAD, 
        #     window_size=6,
        #     shift_size=3,
        #     mlp_ratio=4,
        #     dropout=0.1,
        #     use_gx = use_gx
        # )

        # self.decoder = Decoder(
        #     vocab_size = self.vocab_size, 
        #     embed_dim = cfg.MODEL.BILINEAR.DIM, 
        #     depth = cfg.MODEL.BILINEAR.DECODE_LAYERS,
        #     num_heads = cfg.MODEL.BILINEAR.HEAD, 
        #     dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
        #     ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        #     use_gx = use_gx
        # )


        """Our code"""
        self.att_embed = nn.Sequential(
                nn.Linear(768, 512),
                utils.activation('CELU'),
                nn.Identity(),
                nn.Dropout(0.1)
            )
        
        # I think this acts as a global vector to be added to input feature map
        use_gx = True

        self.encoder = Encoder(
            embed_dim=512, 
            input_resolution=(12, 8), 
            depth=2,        #this can be played with, original value = 3 
            num_heads=5,    #this can be played with, original value = 8
            window_size=4,  #input sent is 768 x 12 x 8
            shift_size=3,
            mlp_ratio=4,
            dropout=0.1,
            use_gx = use_gx
        )
        
        self.decoder = Decoder(
            vocab_size = self.vocab_size, 
            embed_dim = 512, 
            depth = 2,        #this can be played with, original value = 3
            num_heads = 5,    #this can be played with, original value = 8
            dropout = 0.1, 
            ff_dropout = 0.1,
            use_gx = use_gx
        )

    def forward(self, att_feats, seq, gv_feat):#, **kwargs):
        
        """Dimensions of the inputs"""
        # seq is of size B x 95
        # att_feats are directly sent from the swin backbone we trained
        # att_feats are of size B x 768 x 12 x 8
        # gv_feat is just a B x [[0]] tensor

        """PureT code"""
        #att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        #seq = kwargs[cfg.PARAM.INPUT_SENT]
        #import numpy as np
        #seq = np.zeros((1, 5, 17), dtype='int')
        #seq = torch.cat([torch.from_numpy(b) for b in seq], 0).cuda()

        # backbone forward
        #att_feats = self.backbone(att_feats)
        #print(att_feats.shape)
        
        # att_mask for features
        #att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        """the below is taken from PureT dataloader.py"""
        att_mask = torch.ones(att_feats.size()[0], 12 * 8).cuda()
        self.att_feats = att_feats
        self.att_mask = att_mask
        self.gv_feat = gv_feat
        # att_feats = att_feats.cuda()
        # seq = seq.cuda()
        
        """In COCO dataset, each image has 5 captions, so this expansion is needed"""
        #att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        #att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)


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

    def get_logprobs_state(self):#, **kwargs):
        wt = self.param_wt              #kwargs[cfg.PARAM.WT]
        state = self.param_state        #kwargs[cfg.PARAM.STATE]
        encoder_out = self.att_feats    #kwargs[cfg.PARAM.ATT_FEATS]
        
        att_mask = self.att_mask        #kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = self.gv_feat               #kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

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

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = self.att_feats      #kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = self.att_mask        #kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = 5                   #kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        self.att_feats  = encoder_out           #kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        self.gv_feat = gx                       #kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        self.decoder.init_buffer(batch_size)
        for t in range(self.MODEL_SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            self.param_wt = wt                  #kwargs[cfg.PARAM.WT] = wt
            self.param_state = state            #kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state()     #(**kwargs)
            # [B*cur_beam_size, Vocab_size] --> [B, cur_beam_size, Vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # sum of logprob
            # [B, cur_beam_size, Vocab_size]
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # [B, beam_size], [B, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # update buffer
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # expand input
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                # p_att_feats_tmp = []
                # for p_feat in p_att_feats:
                #     p_key, p_value2 = p_feat
                #     p_key = utils.expand_tensor(p_key, beam_size)
                #     p_value2 = utils.expand_tensor(p_value2, beam_size)
                #     p_att_feats_tmp.append((p_key, p_value2))

                self.att_feats = encoder_out        # kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                self.gv_feat = gx                   # kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                self.att_mask = att_mask            # kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.MODEL_SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.MODEL_SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = 5                               #kwargs['BEAM_SIZE']
        greedy_decode = True                        #kwargs['GREEDY_DECODE']
        att_feats = self.att_feats                  #kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = self.att_mask                    #kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)
        
        state = None
        sents = Variable(torch.zeros((batch_size, self.MODEL_SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, self.MODEL_SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        self.att_feats = encoder_out                #kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        self.gv_feat = gx                           #kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        
        # inference word by word
        for t in range(self.MODEL_SEQ_LEN):
            self.param_wt = wt                      #kwargs[cfg.PARAM.WT] = wt
            self.param_state = state                #kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state()   #(**kwargs)
            
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
    
    def flops(self):
        flops = 0
        #flops += self.backbone.flops()
        # self.att_embed
        flops += 768 * 512
        # encoder decoder
        flops += self.encoder.flops()
        flops += self.encoder.flops()
        # flops += self.decoder.flops()
        return flops