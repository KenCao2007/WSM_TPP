import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, MLE_EncoderLayer
import Utils


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    # temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_non_event_mask(seq):
    """ For masking out the non-event time point"""
    len_q = seq.size(1)
    padding_mask = seq.eq(Constants.GRID)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask_thp(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def get_subsequent_mask_ithp(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class thp_Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))
        # )

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD).to('cuda')
        # self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            MLE_EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        # print('time',time.requires_grad)
        tt = time.unsqueeze(-1) / self.position_vec
        mask = torch.zeros_like(tt).bool()
        result = torch.zeros_like(tt)
        mask[..., 0::2] = True

        result += torch.sin(tt)*mask
        result += torch.cos(tt)*~mask
        # print(result.size(),non_pad_mask.size())
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask_thp(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)

        if event_type.max() > 22 or event_type.min() < 0:
            print("wrong event type")
        for name, param in self.event_emb.named_parameters():
            if torch.any(torch.isnan(param.data)):
                print("wrong parameter")

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output



class ithp_Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD).to('cuda')

        self.enc_layer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        
        result = time.unsqueeze(-1) / self.position_vec
        returned = torch.zeros_like(result)
        returned[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        returned[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return returned * non_pad_mask

    def forward(self, type_q, type_kv, time_q, time_kv, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask_ithp(type_kv)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=type_kv, seq_q=type_kv)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask_non_event = get_non_event_mask(type_kv)
        slf_attn_mask_non_event = slf_attn_mask_non_event.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq + slf_attn_mask_non_event).gt(0)

        type_kv_clone = type_kv.clone()
        type_kv_clone[type_kv_clone == -1] = 0 # 

        # 3 kinds of entry cannot be queried to: (1) future entry (2) padding entry (3) non-event entry


        tem_enc_q = self.temporal_enc(time_q, non_pad_mask)
        tem_enc_kv = self.temporal_enc(time_kv, non_pad_mask)
        enc_input_q = self.event_emb(type_q) + tem_enc_q
        enc_input_kv = self.event_emb(type_kv_clone) + tem_enc_kv

        enc_output, _ = self.enc_layer(
                    enc_input_q,
                    enc_input_kv,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask)
        
        return enc_output





class thp_Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=16, d_inner=8,
            n_layers=1, n_head=1, d_k=16, d_v=16, dropout=0.1, opt=None):
        super().__init__()

        self.encoder = thp_Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.name = 'thp'
        self.num_types = num_types
        self.normalize = None
        self.d_inner = d_inner
        self.data_name = opt.data_name
        # self.method = opt.method


        self.base_layer = nn.Sequential(
                nn.Linear(d_model, num_types, bias=True)
                )

        self.affect_layer = nn.Sequential(
                nn.Linear(d_model, num_types, bias=True),
                nn.Tanh()
                )
        self.intensity_layer = nn.Sequential(
                nn.Softplus(beta=1.0)
                )
        
        self.inconsistent_T = opt.inconsistent_T
    
    def forward(self, event_type, event_time, time_gap, opt):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        time_gap = torch.cat((event_time[:,0:1], time_gap), axis = 1)
        event_time = torch.concatenate((torch.zeros(event_time.shape[0], 1).to(opt.device), event_time), axis = 1)
        event_type = torch.cat((torch.ones(event_type.shape[0], 1).type(torch.long).to(opt.device), event_type), axis = 1)
        

        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        self.enc_output = enc_output
        if opt.method == "wsm":
            loss = self.compute_loss_wsm(event_type, event_time, time_gap, non_pad_mask, opt.h_type, opt.CE_coef)
        elif opt.method == "mle":
            loss = self.compute_loss_mle(event_type, event_time, time_gap, non_pad_mask, num_grid = opt.num_grid)
        elif opt.method == "dsm":
            loss = self.compute_loss_dsm(event_type, event_time, time_gap, non_pad_mask, opt.noise_var, opt.CE_coef)
        else:
            raise ValueError("No such method")
        return loss, enc_output


        
    
    def compute_loss_wsm(self, event_type, event_time, time_gap, non_pad_mask, h_type="two_side_op",alpha = 1):
        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)
        
        t_var = torch.autograd.Variable(diff_time, requires_grad=True)

        all_intensity, score = self.get_intensity_n_score(t_var, event_type, event_time, time_gap, non_pad_mask)
        score_grad = torch.autograd.grad(score.sum(), t_var, retain_graph=True)[0]
        sum_intensity  = all_intensity.sum(-1)

        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_time.device)

        type_indices = torch.arange(1, self.num_types + 1, device=event_time.device).view(1, 1, -1)
        type_mask = (event_type.unsqueeze(-1) == type_indices).to(event_time.device)

        type_intensity = (all_intensity * type_mask[:, 1:, :]).sum(-1)
        
        CELoss = -(type_intensity + 1e-10).log() + (sum_intensity + 1e-10).log()

        if self.data_name not in ["exp-decay-multivariate", 'half-sin_multivariate']:
            # max_observed =  torch.max(event_time, axis = 1)[0].unsqueeze(-1)
            if self.inconsistent_T:
                max_observed = max_observed =  torch.max(event_time, axis = 1)[0].unsqueeze(-1)
            else:
                max_observed =  event_time.max()
        elif self.data_name == "exp-decay-multivariate":
            max_observed = 10
        else:
            max_observed = 5
        
        
        if h_type == "two_side_op":
            t_prior = event_time[:,:-1]
            h = (max_observed - t_prior)/2 - abs(event_time[:,1:] - (max_observed + t_prior)/2)
            hprime = torch.where(event_time[:,1:] > (max_observed + t_prior)/2, -1, 1)
        elif h_type == "None":
            # t_prior = event_time[:,:-1]
            # t_next = torch.cat((event_time[:,1:], torch.ones_like(event_time[:,-1:] * max_)))
            h = torch.ones_like(event_time[:,1:])
            hprime = torch.zeros_like(event_time[:,1:])
        elif h_type == "two_side_ord":
            h = (max_observed - event_time[:,1:]) * time_gap
            hprime = max_observed - event_time[:,1:] - time_gap
        elif h_type == "one_side_ord":
            h = time_gap ** 2
            hprime = 2 * time_gap       
        elif h_type == "one_side_opt":
            h = time_gap
            hprime = torch.ones_like(time_gap)
        else:
            raise ValueError("No such h_type")

    
        WSMLoss = (0.5 * h * score ** 2 + score_grad * h + score * hprime)

        loss = ( WSMLoss + alpha * CELoss) * non_pad_mask[:,1:].squeeze(-1)
        return loss
    

    def compute_loss_dsm(self, event_type, event_time, time_gap, non_pad_mask, var_noise = 0.5, alpha = 1, num_noise = 1):
        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)
        
        noise = var_noise * torch.randn([*diff_time.size(), num_noise], device = diff_time.device)
        t_noise = diff_time[:,:,None] + noise
        t_var = t_noise
        t_var = torch.autograd.Variable(t_var, requires_grad=True)

        score = self.get_score(t_var, event_type, event_time, time_gap, non_pad_mask)
        noise_score = -noise / var_noise ** 2

        all_intensity = self.get_intensity(diff_time, event_type, event_time, time_gap, non_pad_mask).squeeze(2)
        sum_intensity  = all_intensity.sum(-1)
        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_time.device)

        type_indices = torch.arange(1, self.num_types + 1, device=event_time.device).view(1, 1, -1)
        type_mask = (event_type.unsqueeze(-1) == type_indices).to(event_time.device)

        type_intensity = (all_intensity * type_mask[:, 1:, :]).sum(-1)
        
        CELoss = -(type_intensity + 1e-10).log() + (sum_intensity + 1e-10).log()

        loss = (0.5 * (score - noise_score) ** 2 * non_pad_mask[:,1:,:]).sum(-1) / num_noise
        loss += alpha * CELoss * non_pad_mask[:,1:].squeeze(-1)

        return loss
    
    def get_intensity_n_score(self, t, event_type, event_time, time_gap, non_pad_mask):
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        self.affect = self.affect_layer(self.enc_output)
        self.base = self.base_layer(self.enc_output)
        
        intensity = self.intensity_layer(self.affect[:,:-1,None,:] * t.unsqueeze(3) + self.base[:,:-1,None,:]).squeeze(3) # (batch*len-1*1/num_samples)
        all_lambda = intensity * non_pad_mask[:, :-1, None, :]
        intensity_total = all_lambda.sum(-1)

        intensity_total_log = ((intensity_total+1e-10).log()*non_pad_mask[:,1:,:])
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t, create_graph=True)[0]*non_pad_mask[:,1:,:]
        score = intensity_total_grad_t - intensity_total

        return all_lambda.squeeze(-2), score.squeeze(-1)
    
            
    def get_intensity(self, t, event_type, event_time, time_gap, non_pad_mask):
        # t size: batch*len-1*num_samples
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        self.affect = self.affect_layer(self.enc_output)
        self.base = self.base_layer(self.enc_output)
        
        intensity = self.intensity_layer(self.affect[:,:-1,None,:] * t.unsqueeze(3) + self.base[:,:-1,None,:]).squeeze(3) # (batch*len-1*1/num_samples)
        all_lambda = intensity * non_pad_mask[:, :-1, None, :]

        
        return all_lambda
    

    def get_score(self, t, event_type, event_time, time_gap, non_pad_mask):
        # t size: batch*len-1*num_samples
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        all_intensity = self.get_intensity(t, event_type, event_time, time_gap, non_pad_mask) # batch*len*num_samples*num_type
        intensity_total = all_intensity.sum(-1)*non_pad_mask[:,1:,:]

        intensity_total_log = ((intensity_total+1e-10).log()*non_pad_mask[:,1:,:])
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t, create_graph=True)[0]*non_pad_mask[:,1:,:]
        score = intensity_total_grad_t - intensity_total

        return score


    def compute_loss_mle(self, event_type, event_time, time_gap, non_pad_mask, num_grid = 10):
        non_pad_mask = get_non_pad_mask(event_type)
            
        event_ll, non_event_ll = Utils.log_likelihood(self, event_time, time_gap, event_type, num_grid)
        event_loss = -torch.sum(event_ll - non_event_ll)
        loss = event_loss
        return loss
    
    def predict(self, event_type, event_time, time_gap, opt):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        time_gap = torch.cat((event_time[:,0:1], time_gap), axis = 1)
        event_time = torch.concatenate((torch.zeros(event_time.shape[0], 1).to(opt.device), event_time), axis = 1)
        event_type = torch.cat((torch.ones(event_type.shape[0], 1).type(torch.long).to(opt.device), event_type), axis = 1)
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        self.enc_output = enc_output
        intensity_pred = self.get_intensity(time_gap, event_type, event_time, time_gap, non_pad_mask).squeeze(2)
        _, type_pred = torch.max(intensity_pred, dim=-1)
        return type_pred + 1
    
    def compute_score(self, event_type, event_time, time_gap):
        # this function is only for debug
        non_pad_mask = get_non_pad_mask(event_type)
        self.enc_output = self.encoder(event_type, event_time, non_pad_mask)
        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)
        
        t_var = torch.autograd.Variable(diff_time, requires_grad=True)

        score = self.get_score(t_var, event_type, event_time, time_gap, non_pad_mask).squeeze(-1)
        score_grad = torch.autograd.grad(score.sum(), t_var, retain_graph=True)[0]

        return score, score_grad
    


    # def compute_fine_grained_intensity(self, event_type, event_time, opt):
        
    #     num_types = opt.num_types # padding & num of events
        
    #     mix_time, mix_type, _ = Utils.grids_events(event_time=event_time, event_type=event_type, opt=opt) # generate mix sequence with grids
    #     non_pad_mask = mix_type[:, 1:] != Constants.PAD
    #     _, enc_dict = self.forward(mix_type, mix_time, None, opt)

    #     intensity = {}
    #     time_gap = (event_time[:, 1:] - event_time[:, :-1])*non_pad_mask[:, 1:]
    #     intensity = self.get_intensity(time_gap, event_type, event_time, time_gap, non_pad_mask.unsqueeze(-1))
    #     for i in range(num_types):
    #         intensity[i] = intensity[:,:,:,i].squeeze(-1)
    #     return intensity, mix_time




class ithp_Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, opt=None):
        super().__init__()

        self.encoder = ithp_Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.intensity_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_v, 1),
                nn.Softplus()
            ) for _ in range(self.num_types)])
            
    
    def compute_intensity(self, event_type, event_time, opt):

        non_pad_mask = get_non_pad_mask(event_type)
        time_gap = torch.concatenate((event_time[:, 0:1], event_time[:, 1:] - event_time[:, :-1]), axis = 1)

        _, enc_dict = self.forward(event_type, event_time, time_gap, opt)
        intensity_learned = {}
        for i in range(opt.num_types):
            intensity_learned[i] = self.intensity_decoder[i](enc_dict[i+1]) * non_pad_mask

        return intensity_learned
    
    def compute_fine_grained_intensity(self, event_type, event_time, opt):
        
        num_types = opt.num_types # padding & num of events
        
        mix_time, mix_type, _ = Utils.grids_events(event_time=event_time, event_type=event_type, opt=opt) # generate mix sequence with grids
        non_pad_mask = mix_type != Constants.PAD
        _, enc_dict = self.forward(mix_type, mix_time, None, opt)

        intensity = {}
        for i in range(num_types):

            intensity_type_i = self.intensity_decoder[i](enc_dict[i+1]) * non_pad_mask.unsqueeze(-1)

            intensity[i]=intensity_type_i.squeeze(-1).detach().cpu().numpy()
        return intensity, mix_time
    
    def forward(self, event_type, event_time, time_gap, opt):

        enc_dict = {}
        if opt.method == "mle":
            event_time, event_type, grid_length = Utils.grids_events(event_time=event_time, event_type=event_type, opt=opt) # generate mix sequence with grids
            time_gap = torch.cat((event_time[:,0:1], (event_time[:,1:] - event_time[:,:-1])*get_non_pad_mask(event_type)[:,1:].squeeze(-1)),axis=-1)
            non_pad_mask = (event_type != Constants.PAD).unsqueeze(-1)
            self.grid_length = grid_length
        else:
            non_pad_mask = get_non_pad_mask(event_type)

        time_kv = event_time.clone()
        time_q = event_time.clone()
        type_kv = event_type.clone()
        if not time_q.requires_grad:
            time_q = torch.autograd.Variable(time_q, requires_grad=True)

        for i in range(1, self.num_types + 1):
            type_q = event_type.clone()
            type_q[type_q != 0] = i  # manually set all entries except padding to be type i.
            enc_output= self.encoder(type_q=type_q, 
                                    type_kv=type_kv, 
                                    time_q=time_q, 
                                    time_kv=time_kv,
                                    non_pad_mask=non_pad_mask)
            enc_dict[i]=enc_output
        self.enc_dict = enc_dict

        if opt.method == "wsm":
            loss = self.compute_loss_wsm(event_type, time_q, time_gap, non_pad_mask)
        elif opt.method == "mle":
            loss = self.compute_loss_mle(event_type, event_time, time_gap, non_pad_mask)
        elif opt.method == "dsm":
            loss = self.compute_loss_dsm(event_type, event_time, time_gap)
        else:
            raise ValueError("No such method")
        
        return loss, self.enc_dict
        
    
    def compute_loss_wsm(self, event_type, time_q, time_gap, non_pad_mask, alpha = 1):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        

        if not time_q.requires_grad:
            time_q = torch.autograd.Variable(time_q, requires_grad=True)

        overall_intensity = torch.zeros_like(time_q)
        log_intensity = torch.zeros_like(time_q)
        for i in range(1, self.num_types + 1):
            enc_output = self.enc_dict[i]
            intensity_type_i = self.intensity_decoder[i-1](enc_output)*non_pad_mask  # Generate the intensity at type i
            overall_intensity += intensity_type_i.squeeze(-1)

            log_intensity_type_i = (torch.log(intensity_type_i+1e-5)*non_pad_mask).squeeze(-1)
            mask_type_i = event_type == i # choose all positions where type i occurs
            log_intensity[mask_type_i] = log_intensity_type_i[mask_type_i]

        log_overall_intensity = torch.log(overall_intensity+1e-5)*non_pad_mask.squeeze(-1)
        score = torch.autograd.grad(log_overall_intensity.sum(), time_q, create_graph=True)[0]-overall_intensity.squeeze(-1)
        score_grad = torch.autograd.grad(score.sum(), time_q,  create_graph=True)[0]
        
        max_observed =  torch.max(time_q)    
        t_prior = torch.cat((torch.zeros(time_q.shape[0], 1).to(time_q.device), time_q[:, :-1]), dim=1)
        h = (max_observed - t_prior)/2 - abs(time_q - (max_observed + t_prior)/2)
        hprime = torch.where(time_q > (max_observed + t_prior)/2, -1, 1)

        loss = (0.5 * h * score ** 2 + score_grad * h + score * hprime) * non_pad_mask.squeeze(-1)
        loss -= alpha * (log_intensity - log_overall_intensity) * non_pad_mask.squeeze(-1)

        return loss
    


    def compute_loss_mle(self, mix_type, event_time, time_gap, non_pad_mask):
        log_event_ll = 0
        non_event_ll = 0

        enc_dict = self.enc_dict
        intensity = {}
        for i in range(self.num_types):

            _type_i_mask = (mix_type == i).unsqueeze(-1)

            _grids_mask = (mix_type == Constants.GRID).unsqueeze(-1)

            intensity_type_i = self.intensity_decoder[i](enc_dict[i+1]) #* non_pad_mask.unsqueeze(-1)

            log_intensity_type_i = torch.log(intensity_type_i+1e-5) * non_pad_mask

            log_intensity_event = log_intensity_type_i * _type_i_mask  # Select event intensity

            grid_intensity = intensity_type_i *  _grids_mask  # Select non-event intensity

            log_event_ll += torch.sum(log_intensity_event)
            non_event_ll += torch.sum(grid_intensity * self.grid_length)

            intensity[i]=intensity_type_i.squeeze(-1).detach().cpu().numpy()

        tll = log_event_ll - non_event_ll

        return tll
        

    
    def predict(self, event_type, event_time, time_gap, opt):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        # self.forward(event_type, event_time, time_gap, opt)
        intensity_pred = self.compute_intensity(event_type, event_time, opt)
        intensity_pred = torch.stack(list(intensity_pred.values()), dim=-1).squeeze(2)[:,1:]
        _, type_pred  = torch.max(intensity_pred, dim=-1)
        return type_pred + 1







