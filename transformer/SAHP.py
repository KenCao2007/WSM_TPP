import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Constants
import math

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


def get_end_mask(mask):
     # mask: (batch_size, seq_len)
    # This function returns a nonmask label, if 1 represents this is the end of the sequence
    
    device = mask.device
    row_sum = mask.sum(dim=1).squeeze().to(int)  # (batch_size)
    end_mask = torch.zeros(mask.shape[0], mask.shape[1] + 1, 1).to(device)
    end_mask[torch.arange(end_mask.size(0), device=end_mask.device), row_sum , 0] = 1 # (batch_size, seq_len)
    return end_mask

def get_non_event_mask(seq):
    """ For masking out the non-event time point"""
    len_q = seq.size(1)
    padding_mask = seq.eq(Constants.GRID)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

class SublayerConnection(nn.Module):
    # used for residual connection
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class TimeShiftedPositionalEncoding(nn.Module):
    """Time shifted positional encoding in SAHP, ICML 2020
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # [max_len, 1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # [model_dim //2 ]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        self.layer_time_delta = nn.Linear(1, d_model // 2, bias=False)

        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

    def forward(self, x, interval):
        """

        Args:
            x: time_seq, [batch_size, seq_len]
            interval: time_delta_seq, [batch_size, seq_len]

        Returns:
            Time shifted positional encoding defined in Equation (8) in SAHP model

        """
        phi = self.layer_time_delta(interval.unsqueeze(-1))
        
        if len(x.size()) > 1:
            length = x.size(1)
        else:
            length = x.size(0)

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x_q, x_kv, mask):
        return self.self_attn(x_q, x_kv, x_kv, mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False, d_k=None, d_v=None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_k if d_k is not None else d_model // n_head
        self.d_v = d_v if d_v is not None else self.d_k
        self.d_model = d_model
        self.output_linear = output_linear


        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]

        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            if output_weight:
                return self.linears[-1](x), attn_weight
            else:
                return self.linears[-1](x)
        else:
            if output_weight:
                return x, attn_weight
            else:
                return x

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # small change here -- we use "1" for masked element
        scores = scores.masked_fill(mask > 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn




class SAHP(nn.Module):
    """Torch implementation of Self-Attentive Hawkes Process, ICML 2020.
    Part of the code is collected from https://github.com/yangalan123/anhp-andtt/blob/master/sahp

    I slightly modify the original code because it is not stable.

    """

    def __init__(self, model_config, max_time):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super().__init__()
        self.d_model = model_config.d_model
        self.d_time = model_config.d_model
        self.num_types = model_config.num_types
        self.d_inner = model_config.d_inner_hid
        self.use_norm = False
        self.max_time = max_time
        # position vector, used for temporal encoding
        self.layer_position_emb = TimeShiftedPositionalEncoding(d_model=self.d_model)
        # self.layer_position_emb = self.temporal_enc# (d_model=self.d_model)

        self.n_layers = model_config.n_layers
        self.n_head = model_config.n_head
        self.dropout = model_config.dropout 

        # convert hidden vectors into a scalar
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_types)
        self.softplus = nn.Softplus()
        self.with_survival = model_config.with_survival
        self.alpha_neg = model_config.alpha_neg
        self.alpha_survival = model_config.alpha_survival
        self.num_noise = model_config.num_noise
        self.noise_var = model_config.noise_var
        self.CE_coef = model_config.CE_coef
        self.noise_type = model_config.noise_type

        if not self.with_survival:
            self.alpha_survival = 0.0


        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)

        # Equation (12): mu
        self.mu = nn.Sequential(
            nn.Linear(self.d_model, self.num_types),
            # nn.GELU()
            )
        #nn.Parameter(torch.empty([self.d_model, self.num_types])).to('cuda')
        # Equation (13): eta
        self.eta = nn.Sequential(
            nn.Linear(self.d_model, self.num_types),
            # nn.GELU()
            )
        # Equation (14): gamma
        self.gamma = nn.Sequential(
            nn.Linear(self.d_model, self.num_types),
            # nn.Softplus()
            )
        
        nn.init.xavier_normal_(self.mu[0].weight)
        nn.init.xavier_normal_(self.eta[0].weight)
        nn.init.xavier_normal_(self.gamma[0].weight)

        self.layer_type_emb = nn.Embedding(self.num_types+1,  # have padding
                                           self.d_model,
                                           padding_idx=Constants.PAD)

        self.eps = torch.finfo(torch.float32).eps
        self.encoder = EncoderLayer(
                self.d_model,
                MultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                                   output_linear=False),

                use_residual=False,
                dropout=self.dropout
            )

        # self.position_vec = torch.tensor(
        #     [math.pow(10000.0, 2.0 * (i // 2) / self.d_model) for i in range(self.d_model)],
        #     device=torch.device('cuda'))


        self.Survival = nn.Sequential(
            nn.Linear(self.d_model, self.d_inner),
            nn.ReLU(),
            nn.Linear(self.d_inner, 2)
        )
        
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
    
    def compute_intensity(self, batch,  opt):
        type_seqs, time_seqs, time_delta_seqs= batch
        type_seqs = type_seqs.long()
        _, enc_out = self.forward(type_seqs, time_seqs, time_delta_seqs, opt)
        # time_delta_seqs = torch.cat((torch.zeros_like(time_seqs[:,0:1]), (time_seqs[:,1:] - time_seqs[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)
        time_delta_seqs = torch.cat((time_seqs[:,0:1], time_delta_seqs), axis = 1)
        time_delta_seqs = torch.cat((torch.zeros_like(time_seqs[:,0:1]), time_delta_seqs),axis=-1)
        cell_t = self.state_decay(encode_state=enc_out,
                                  duration_t=time_delta_seqs[:, :, None]).squeeze(-2)

        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.softplus(cell_t)
        return lambda_at_event
    # def state_decay(self, encode_state, mu, eta, gamma, duration_t):
    def state_decay(self, encode_state, duration_t):
        """Equation (15), which computes the pre-intensity states

        Args:
            encode_state (tensor): [batch_size, seq_len, hidden_size].
            mu (tensor): [batch_size, seq_len, hidden_size].
            eta (tensor): [batch_size, seq_len, hidden_size].
            gamma (tensor): [batch_size, seq_len, hidden_size].
            duration_t (tensor): [batch_size, seq_len, num_sample].

        Returns:
            tensor: hidden states at event times.
        """


        states = self.mu(encode_state[:,:,None]) + (
            # self.eta(encode_state[:,:, None]) - self.mu(encode_state[:,:-1])) * torch.exp(
            self.eta(encode_state[:,:,None]) - self.mu(encode_state[:,:,None])) * F.softplus(
         -self.gamma(encode_state[:,:,None])* duration_t.unsqueeze(-1))

        return states

    def get_cond(self, t, event_time, event_type, non_pad_mask):
        '''

        Args:
            t (tensor): [batch_size, seq_len], time_gap, starting from x_1 - 0
            event_time (tensor): [batch_size, seq_len + 1], event time seqs. starting from x_0 = 0
            event_type (tensor): [batch_size, seq_len + 1], event type seqs
            non_pad_mask (tensor): [batch_size, seq_len + 1], sequence mask vector to mask the padded events.
        Returns:
            tensor: [batch_size, seq_len, hidden_size], hidden states at event times.
        '''
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0) 

        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)

        type_embedding = self.layer_type_emb(event_type)

        enc_output = type_embedding + self.layer_position_emb(event_time, time_gap)
        enc_output = self.encoder(
                enc_output,
                enc_output,
                mask=slf_attn_mask
                )
        self.enc_output = enc_output
        return enc_output

    def get_survival_loss(self, non_pad_mask, mode = "eval", alpha_neg=1.0, focal_gamma=0.0, eps=1e-12):
        end_mask = get_end_mask(non_pad_mask[:, 1:]).squeeze(-1).long()          # y∈{0,1}
        survival_pred = self.Survival(self.enc_output)                           # logits

        valid = non_pad_mask.squeeze(-1).bool()
        logits = survival_pred[valid]                                            # (N, 2)
        y      = end_mask[valid]                                                 # (N,)

        ce = F.cross_entropy(logits, y, reduction='none')                        # (N,)

        if mode == "train":
            if alpha_neg != 1.0:
                w_neg = torch.full_like(ce, alpha_neg)
                w_pos = torch.ones_like(ce)
                w = torch.where(y == 0, w_neg, w_pos)                            # (N,)
                ce = ce * w

            if focal_gamma and focal_gamma > 0.0:
                pt = logits.softmax(dim=-1).gather(1, y.view(-1, 1)).squeeze(1)  # (N,)
                mod = (1.0 - pt + eps).pow(focal_gamma)
                ce = ce * mod

        survival_loss = ce.sum()
        return survival_loss

    def forward(self, event_type, event_time, time_gap, opt):
        """
        starts from 0, fit t_1,...., t_N   
        """
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.get_cond(time_gap, event_time, event_type, non_pad_mask)

        if opt.method == "dsm":
           loss = self.compute_loss_dsm(event_type[:,1:], time_gap, non_pad_mask, opt.noise_var, opt.CE_coef, opt.num_noise, opt.alpha_survival)
        elif opt.method == "mle":
            loss = self.compute_loss_mle(event_type, event_time, time_gap, num_grid = opt.num_grid)
        elif opt.method == "wsm":
            loss = self.compute_loss_wsm(event_type[:, 1:], event_time, time_gap, non_pad_mask, opt.h_type, opt.CE_coef)
        else:
            raise ValueError("No such method")
        return loss, enc_output
    
    def compute_loss_wsm(self, event_type, event_time, time_gap, non_pad_mask, h_type, alpha = 1):
         # make sure the shape of event_type, time_gap, non_pad_mask are the same (for the first two dimensions)
        
        t_var = torch.autograd.Variable(time_gap, requires_grad=True)

        all_intensity, score = self.get_intensity_n_score(t_var, self.enc_output[:, :-1], non_pad_mask[:, 1:])
        score_grad = torch.autograd.grad(score.sum(), t_var, create_graph=True, retain_graph=True)[0]
        sum_intensity  = all_intensity.sum(-1)

        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_type.device)

        type_indices = torch.arange(1, self.num_types + 1, device=event_type.device).view(1, 1, -1)
        type_mask = (event_type.unsqueeze(-1) == type_indices).to(event_type.device)

        type_intensity = (all_intensity * type_mask).sum(-1)

        
        CELoss = -(type_intensity + 1e-10).log() + (sum_intensity + 1e-10).log()

        max_observed = self.max_time
        
        
        # if h_type == "two_side_op":
        t_prior = event_time[:,:-1]
        h = (max_observed - t_prior)/2 - abs(event_time[:,1:] - (max_observed + t_prior)/2)
        hprime = torch.where(event_time[:,1:] > (max_observed + t_prior)/2, -1, 1)
        # elif h_type == "None":
        #     h = torch.ones_like(event_time[:,1:])
        #     hprime = torch.zeros_like(event_time[:,1:])
        # elif h_type == "two_side_ord":
        #     h = (max_observed - event_time[:,1:]) * time_gap
        #     hprime = max_observed - event_time[:,1:] - time_gap
        # elif h_type == "one_side_ord":
        #     h = time_gap ** 2
        #     hprime = 2 * time_gap       
        # elif h_type == "one_side_opt":
        #     h = time_gap
        #     hprime = torch.ones_like(time_gap)
        # else:
        #     raise ValueError("No such h_type")

        survival_loss = self.get_survival_loss(non_pad_mask, mode = "train", alpha_neg = self.alpha_neg)

        # end_mask = get_end_mask(non_pad_mask[:, 1:]).squeeze(-1).long()
        # survival_pred = self.Survival(self.enc_output)
        # valid = non_pad_mask.squeeze(-1).bool()
        # survival_loss = nn.CrossEntropyLoss()(survival_pred[valid],  end_mask[valid])
        # return survival_loss

        WSMLoss = (0.5 * h * score ** 2 + score_grad * h + score * hprime) * non_pad_mask[:,1:].squeeze(-1)

        loss = (( WSMLoss + alpha * CELoss) * non_pad_mask[:,1:].squeeze(-1)).sum() + survival_loss * self.alpha_survival
        return loss
    

    def compute_loss_dsm(self, event_type, time_gap, non_pad_mask, var_noise = 0.5, alpha = 1, num_noise = 1, alpha_survival = 1.):
        # raise NotImplementedError("This method is not implemented yet")
        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)
        
        noise = var_noise * torch.randn([*diff_time.size(), num_noise], device = diff_time.device)
        if self.noise_type == "normal":
            t_noise = diff_time[:,:,None] + noise
            t_var = t_noise
            noise_score = -noise / var_noise ** 2
        elif self.noise_type == "lognormal":
            t_noise = diff_time[:,:,None] * torch.exp(noise)
            t_var = t_noise
            noise_score = -1/ (t_var + 1e-10) * (1 + noise / var_noise ** 2)

        t_var = torch.autograd.Variable(t_var, requires_grad=True)

        all_intensity, score = self.get_intensity_n_score(t_var, self.enc_output[:, :-1], non_pad_mask[:, 1:])
        

        all_intensity = self.get_intensity(diff_time, self.enc_output[:, :-1], non_pad_mask[:, 1:]).squeeze(2)
        sum_intensity  = all_intensity.sum(-1)
        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_type.device)

        type_indices = torch.arange(1, self.num_types + 1, device=event_type.device).view(1, 1, -1)
        type_mask = (event_type.unsqueeze(-1) == type_indices).to(event_type.device)

        type_intensity = (all_intensity * type_mask).sum(-1)
        
        CELoss = -(type_intensity + 1e-10).log() + (sum_intensity + 1e-10).log()
        if self.noise_type == "normal":
            loss = (0.5 * (score - noise_score) ** 2 * non_pad_mask[:,1:,:]).sum(-1) / num_noise * var_noise ** 2
        elif self.noise_type == "lognormal":
            loss = (0.5 * (score - noise_score) ** 2 * var_noise ** 2 * t_var.detach() * non_pad_mask[:,1:,:]).sum(-1) / num_noise
        # loss *= var_noise ** 2
        loss += alpha * CELoss * non_pad_mask[:,1:].squeeze(-1)

        loss = loss.sum()
        # add survival loss
        if self.with_survival == 1:
            survival_loss = self.get_survival_loss(non_pad_mask, mode = "train", alpha_neg = self.alpha_neg)
            loss += survival_loss * alpha_survival

        return loss
    
    def get_intensity(self, t, cond, non_pad_mask):
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3
        cell_t = self.state_decay(encode_state=cond,
                            duration_t=t)
        all_intensity = self.softplus(cell_t) * non_pad_mask.unsqueeze(-1)
        return all_intensity
    


        
        
    def get_intensity_n_score(self, t, cond, non_pad_mask):

        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3
        cell_t = self.state_decay(encode_state=cond,
                            duration_t=t)
        all_intensity = self.softplus(cell_t)
        intensity_total = all_intensity.sum(-1) * non_pad_mask

        intensity_total_log = ((intensity_total+1e-10).log())
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t, create_graph=True)[0] * non_pad_mask
        score = (intensity_total_grad_t - intensity_total)
        # score_grad = torch.autograd.grad(score.sum(), t, create_graph=True)[0][:,1:] * non_pad_mask[:,1:]

        # return all_intensity, score, score_grad
        return all_intensity.squeeze(-2), score.squeeze(-1)

    def compute_event(self, time_gap, event_type, non_pad_mask):
        # [batch_size, seq_len-1, hidden_dim]
        lambda_at_event = self.get_intensity(time_gap, self.enc_output[:,:-1], non_pad_mask).squeeze(-2)
        
        # [batch_size, seq_len, hidden_dim]
        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_type.device)
        for i in range(self.num_types):
            type_mask[:, :, i] = (event_type == i + 1).bool().to(event_type.device)

        event = torch.sum(lambda_at_event * type_mask, dim=2)
        event += math.pow(10, -9)
        event.masked_fill_(~non_pad_mask.squeeze(2).bool(), 1.0)
        result = torch.log(event+1e-10) * non_pad_mask.squeeze(2)
        return result


    def compute_integral_unbiased(self, time_gap, cond, non_pad_mask, num_grid):
        num_samples = num_grid
        time_low = 0
        # temp_time = (time_gap.unsqueeze(2) - time_low) * \
        #             torch.rand([*time_gap.size(), num_samples], device=time_gap.device) + time_low
        ratios = (torch.arange(num_samples, device=time_gap.device).float() + 0.5) / num_samples
        temp_time = time_low + (time_gap.unsqueeze(-1) - time_low) * ratios
        
        if self.num_types >= 100:
            all_lambda = None
            for i in range(num_samples):
                lambda_i = self.get_intensity(temp_time[:,:,i:i+1], cond, non_pad_mask)
                if all_lambda == None:
                    all_lambda = torch.sum(lambda_i, dim=(2,3)) 
                else:
                    all_lambda += torch.sum(lambda_i, dim=(2,3)) 
            all_lambda /= num_samples
        else:
            all_lambda = self.get_intensity(temp_time, cond, non_pad_mask)
            all_lambda = torch.sum(all_lambda, dim=(2,3)) / num_samples

        unbiased_integral = all_lambda * (time_gap - time_low) * non_pad_mask.squeeze(-1)
        return unbiased_integral
        
    def compute_loss_mle(self, event_type, event_time, time_gap, num_grid=10):

        ll = self.log_likelihood(event_time, time_gap, event_type, num_grid)
        loss = -ll.sum()
        return loss
    
    def log_likelihood(self, event_time, time_gap, event_type, num_grid):
        
        non_pad_mask = get_non_pad_mask(event_type)
        end_mask = get_end_mask(non_pad_mask)[:, 1:]

        # event log-likelihood
        tilde_lambda_log = self.compute_event(time_gap, event_type[:, 1:], non_pad_mask[:, 1:])
        # non-event log-likelihood, either numerical integration or MC integration
        
        if not self.with_survival:
            event_time_end = torch.cat((event_time[:,1:], torch.zeros(event_time.shape[0], 1).to(event_time.device)), axis = 1)
            event_time_end += (non_pad_mask.squeeze(-1) - torch.concat((non_pad_mask[:,1:].squeeze(-1), torch.zeros(non_pad_mask.shape[0], 1).to(non_pad_mask.device)), axis = 1)) * self.max_time
            time_gap_expand = torch.cat((event_time_end[:, :1], event_time_end[:, 1:] - event_time_end[:, :-1]), axis = 1) * non_pad_mask.squeeze()
            tilde_Lambda_n = self.compute_integral_unbiased(time_gap_expand, self.enc_output, non_pad_mask, num_grid)
            ll = (tilde_lambda_log * non_pad_mask[:, 1:].squeeze(-1)).sum(-1) - (tilde_Lambda_n * non_pad_mask.squeeze(-1)).sum(-1)
            # ll = ((tilde_lambda_log - tilde_Lambda_n) * non_pad_mask[:, 1:].squeeze(-1)).sum(-1)
        else:
            tilde_Lambda_n = self.compute_integral_unbiased(time_gap, self.enc_output[:, :-1], non_pad_mask[:, 1:], num_grid)
            time_gap_end = (self.max_time - event_time[:, :-1]) * non_pad_mask[:, 1:].squeeze(-1)
            tilde_Lambda_T = self.compute_integral_unbiased(time_gap_end, self.enc_output[:, :-1], non_pad_mask[:, 1:], num_grid)
            tilde_G_n_T = torch.exp(-tilde_Lambda_T) * non_pad_mask[:, 1:].squeeze(-1)


            hat_G_n = torch.sigmoid(self.Survival(self.enc_output)[:, :, 0:1]) * (1 - end_mask) * non_pad_mask
            hat_G_n += torch.sigmoid(self.Survival(self.enc_output)[:,:, 1:]) * end_mask 

            ll = ((tilde_lambda_log - tilde_Lambda_n - (1 - tilde_G_n_T + 1e-10).log()) * non_pad_mask[:, 1:].squeeze(-1)).sum(-1)\
                + ((hat_G_n + 1e-10).log() * non_pad_mask).squeeze(-1).sum(-1)

        return ll.sum()
        

    def make_dtime_loss_samples(self, time_delta_seq, num_grid):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=num_grid)[None, None, :].to('cuda')

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes


    def compute_states_at_sample_times(self,
                                       encode_state,
                                       sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            encode_state (tensor): three tensors with each shape [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: [batch_size, seq_len, num_samples, hidden_size], hidden state at each sampled time.
        """
        
        # cell_states = self.state_decay(encode_state[:, :, None, :],
        #                                sample_dtimes[:, :, :, None])

        
        cell_states = self.mu(encode_state[:,:, None, :]) + (
            # self.eta(encode_state[:,:-1, None, :]) - self.mu(encode_state[:,:-1, None, :])) * torch.exp(
            self.eta(encode_state[:,:, None, :]) - self.mu(encode_state[:,:, None, :])) * F.softplus(
         -self.gamma(encode_state[:,:, None, :]) * sample_dtimes[:,:,:,None])

        return cell_states

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask,
                              lambda_type_mask):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            lambda_type_mask (tensor): [batch_size, seq_len, num_event_types], type mask matrix to mask the
            padded event types.

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """

        # Sum of lambda over every type and every event point
        # [batch_size, seq_len]
        event_lambdas = torch.sum(lambda_at_event * lambda_type_mask, dim=-1) + self.eps
        # mask the pad event
        event_lambdas = event_lambdas.masked_fill_(~seq_mask, 1.0)

        # [batch_size, seq_len)
        event_ll = torch.log(event_lambdas)
        # [batch_size, seq_len, n_loss_sample]

        lambdas_total_samples = lambdas_loss_samples.sum(dim=-1)

        # interval_integral - [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        non_event_ll = lambdas_total_samples.mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]

        return event_ll, non_event_ll, num_events

    def predict(self, event_type, event_time, time_gap, opt=None):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        # time_gap = torch.cat((event_time[:,0:1], time_gap), axis = 1)
        # event_time = torch.concatenate((torch.zeros(event_time.shape[0], 1).to(opt.device), event_time), axis = 1)
        # event_type = torch.cat((torch.ones(event_type.shape[0], 1).type(torch.long).to(opt.device), event_type), axis = 1)
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.get_cond(time_gap, event_time, event_type, non_pad_mask)
        self.enc_output = enc_output
        intensity_pred = self.get_intensity(time_gap, self.enc_output[:, :-1], non_pad_mask[:, 1:]).squeeze(2)
        _, type_pred = torch.max(intensity_pred, dim=-1)
        return type_pred + 1
    
    def eval_intensity(self, event_type, event_time, time_gap, num_grid=10):

        #evaluate intensity function
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.get_cond(time_gap, event_time, event_type, non_pad_mask)
        self.enc_output = enc_output

        temp_time = (time_gap.unsqueeze(2) - 0.) * \
                    torch.rand([*time_gap.size(), num_grid], device=time_gap.device) + 0.
        temp_tilde_lambda = self.get_intensity(temp_time, enc_output[:, :-1], non_pad_mask[:, 1:])  # batch*seq_len-1*num_grid*num_type

        # flatten the temp_time, such that the dimension is (batch*num_grid, seq_len). The different num_grid are piled up in the batch dimension
        time_gap_flatten = temp_time.permute(2,0,1).contiguous().view(-1, temp_time.shape[1])
        cond_aug = enc_output[:,:-1,:,None].repeat(1,1,1,num_grid).permute(3,0,1,2).contiguous().view(-1, time_gap.shape[1], enc_output.shape[2])
        non_pad_mask_aug = non_pad_mask[:,1:,:,None].repeat(1,1,1,num_grid).permute(3,0,1,2).contiguous().view(-1, time_gap.shape[1], non_pad_mask.shape[2])
        tilde_Lambda_n_aug = self.compute_integral_unbiased(time_gap_flatten, cond_aug, non_pad_mask_aug, num_grid=200)
        temp_tilde_Lambda_n =  tilde_Lambda_n_aug.view(num_grid, -1, tilde_Lambda_n_aug.shape[1]).permute(1,2,0).contiguous()  # batch*seq_len-1*num_grid
        temp_tilde_G_n = (torch.exp(-temp_tilde_Lambda_n) * non_pad_mask[:, 1:]).unsqueeze(-1) # batch*seq_len-1*num_grid*1

        time_gap_end = (self.max_time - event_time[:, :-1]) * non_pad_mask[:, 1:].squeeze(-1)
        tilde_Lambda_T = self.compute_integral_unbiased(time_gap_end, self.enc_output[:, :-1], non_pad_mask[:, 1:], num_grid=200)
        tilde_G_n_T = torch.exp(-tilde_Lambda_T) * non_pad_mask[:, 1:].squeeze(-1) #batch * seq_len-1
        temp_tilde_G_n_T = tilde_G_n_T[:,:, None].repeat(1,1,num_grid).unsqueeze(-1) # batch*seq_len-1*num_grid*1

        hat_F_n = torch.sigmoid(self.Survival(self.enc_output[:, :-1])[:,:, 0]) * non_pad_mask[:, 1:].squeeze(-1) #batch * seq_len-1
        temp_hat_F_n = hat_F_n[:,:, None].repeat(1,1,num_grid).unsqueeze(-1)  # batch*seq_len-1*num_grid*1

        if not self.with_survival:
            temp_intensity = temp_tilde_lambda 
        else:
            temp_intensity = temp_tilde_G_n * temp_tilde_lambda / (temp_tilde_G_n + ((1-temp_tilde_G_n_T) / (temp_hat_F_n + 1e-10)) - 1 + 1e-10)
        temp_intensity *= non_pad_mask[:, 1:,:,None]
        
        # make negative value in temp_intensity to zero
        # temp_intensity[temp_intensity < 0] = 0.0

        temp_timestamp = temp_time + event_time[:, :-1,None]  # batch*seq_len-1*num_grid

        return temp_intensity, temp_timestamp
    

    