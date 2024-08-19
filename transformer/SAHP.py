import torch
import torch.nn as nn
from transformer.Layers import EncoderLayer
from transformer.torch_basemodel import TorchBaseModel
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
        # if self.use_residual:
        #     x = self.sublayer[0](x, lambda x: self.self_attn(x_q, x_kv, x_kv, mask))
        #     if self.feed_forward is not None:
        #         return self.sublayer[1](x, self.feed_forward)
        #     else:
        #         return x
        # else:
        return self.self_attn(x_q, x_kv, x_kv, mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear

        # self.v_linear = nn.Linear(d_input, d_model)

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
        # print(query.shape, key.shape, value.shape, 'before')
        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]

        # value = self.v_linear(value).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)

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

class mle_SAHP(nn.Module):
    """Torch implementation of Self-Attentive Hawkes Process, ICML 2020.
    Part of the code is collected from https://github.com/yangalan123/anhp-andtt/blob/master/sahp

    I slightly modify the original code because it is not stable.

    """

    def __init__(self, model_config):
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

        # position vector, used for temporal encoding
        self.layer_position_emb = TimeShiftedPositionalEncoding(d_model=self.d_model)
        # self.layer_position_emb = self.temporal_enc# (d_model=self.d_model)

        self.n_layers = model_config.n_layers
        self.n_head = model_config.n_head
        self.dropout = model_config.dropout 

        # convert hidden vectors into a scalar
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_types)
        self.softplus = nn.Softplus()


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
    
    def compute_intensity(self, batch, opt):
        type_seqs, time_seqs, time_delta_seqs= batch
        type_seqs = type_seqs.long()
        _, enc_out = self.forward(type_seqs, time_seqs, time_delta_seqs, opt)

        cell_t = self.state_decay(encode_state=enc_out,
                                  duration_t=time_delta_seqs[:, :, None])

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

        #print(encode_state.shape, duration_t.shape, 'two shape!!')

        states = self.mu(encode_state[:,:-1]) + (
            # self.eta(encode_state[:,:-1]) - self.mu(encode_state[:,:-1])) * torch.exp(
            self.eta(encode_state[:,:-1]) - self.mu(encode_state[:,:-1])) * F.softplus(
         -self.gamma(encode_state[:,:-1]) * duration_t[:,1:])

        return states

    def forward(self, event_type, event_time, time_gap, opt):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            event_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """

        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0) 
        non_pad_mask = get_non_pad_mask(event_type)

        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)

        type_embedding = self.layer_type_emb(event_type)

        enc_output = type_embedding + self.layer_position_emb(event_time, time_gap)
        enc_output = self.encoder(
                enc_output,
                enc_output,
                mask=slf_attn_mask
                )
        
        # [batch_size, seq_len-1, hidden_dim]
        cell_t = self.state_decay(encode_state=enc_output[:,:],
                            duration_t=time_gap[:, :, None])

        # [batch_size, seq_len-1, num_types]
        lambda_at_event = self.softplus(cell_t)

        # print(lambda_at_event.shape, 'the lambda at event, should be [batch_size, *, num_types]')

        # [batch_size, seq_len, hidden_dim]
        type_mask = 0
        for type_ in range(1, self.num_types+1):
            type_i_mask = (event_type[:,:] == type_).unsqueeze(-1)
            if type_==1:
                type_mask = type_i_mask
            else:
                type_mask = torch.cat((type_mask, type_i_mask), axis=-1)

        assert lambda_at_event.shape[1] == cell_t.shape[1] == event_time.shape[1]-1 == time_gap.shape[1]-1

        # Compute the big lambda integral in equation (8) of NHP paper
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length

        # [batch_size, seq - 1, num_sample, event_num]
        sample_dtimes = self.make_dtime_loss_samples(time_gap)

        # print(sample_dtimes.shape, 'the sampled duration shape, should be [batch_size, seq_len, num_sample]')

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(encode_state=enc_output, sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        # print(lambda_at_event.shape, 'event inten!')
        # print(lambda_t_sample.shape, 'non-event inten!')
        

        non_pad_mask = non_pad_mask.squeeze(-1).bool()

        assert lambda_at_event.shape[1] == lambda_t_sample.shape[1] == time_gap.shape[1]-1 == non_pad_mask.shape[1]-1 == type_mask.shape[1]-1
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_gap[:, 1:],
                                                                        seq_mask=non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # return enc_inten to compute accuracy
        log_likelihood = (event_ll - non_event_ll).sum()

        return -log_likelihood/num_events, enc_output

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=20)[None, None, :].to('cuda')

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
            tensor: [batch_size, seq_len, num_samples, hidden_size]， hidden state at each sampled time.
        """
        
        # cell_states = self.state_decay(encode_state[:, :, None, :],
        #                                sample_dtimes[:, :, :, None])

        cell_states = self.mu(encode_state[:,:-1, None, :]) + (
            # self.eta(encode_state[:,:-1, None, :]) - self.mu(encode_state[:,:-1, None, :])) * torch.exp(
            self.eta(encode_state[:,:-1, None, :]) - self.mu(encode_state[:,:-1, None, :])) * F.softplus(
         -self.gamma(encode_state[:,:-1, None, :]) * sample_dtimes[:,1:,:,None])

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

        intensity_pred = self.compute_intensity([event_type, event_time,time_gap], opt)
        # intensity_pred = torch.stack(list(intensity_pred.values()), dim=-1).squeeze(2)[:,1:]
        _, type_pred  = torch.max(intensity_pred, dim=-1)

        return type_pred + 1
    



class SAHP(nn.Module):
    """Torch implementation of Self-Attentive Hawkes Process, ICML 2020.
    Part of the code is collected from https://github.com/yangalan123/anhp-andtt/blob/master/sahp

    I slightly modify the original code because it is not stable.

    """

    def __init__(self, model_config):
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

        # position vector, used for temporal encoding
        self.layer_position_emb = TimeShiftedPositionalEncoding(d_model=self.d_model)
        # self.layer_position_emb = self.temporal_enc# (d_model=self.d_model)

        self.n_layers = model_config.n_layers
        self.n_head = model_config.n_head
        self.dropout = model_config.dropout 

        # convert hidden vectors into a scalar
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_types)
        self.softplus = nn.Softplus()


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

        self.data_name = model_config.data_name
        self.inconsistent_T = model_config.inconsistent_T
        
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

        #print(encode_state.shape, duration_t.shape, 'two shape!!')

        states = self.mu(encode_state[:,:-1]) + (
            # self.eta(encode_state[:,:-1]) - self.mu(encode_state[:,:-1])) * torch.exp(
            self.eta(encode_state[:,:-1]) - self.mu(encode_state[:,:-1])) * F.softplus(
         -self.gamma(encode_state[:,:-1])* duration_t[:,1:])

        return states

    def forward(self, event_type, event_time, time_gap, opt):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            event_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        time_gap = torch.cat((event_time[:,0:1], time_gap), axis = 1)
        event_time = torch.concatenate((torch.zeros(event_time.shape[0], 1).to(opt.device), event_time), axis = 1)
        event_type = torch.cat((torch.ones(event_type.shape[0], 1).type(torch.long).to(opt.device), event_type), axis = 1)


        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0) 
        non_pad_mask = get_non_pad_mask(event_type)

        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)

        type_embedding = self.layer_type_emb(event_type)

        enc_output = type_embedding + self.layer_position_emb(event_time, time_gap)
        enc_output = self.encoder(
                enc_output,
                enc_output,
                mask=slf_attn_mask
                )
        
        self.enc_output = enc_output

        if opt.method == "dsm":
           loss = self.compute_loss_dsm(event_type, event_time, time_gap, non_pad_mask, opt.noise_var, opt.CE_coef, opt.num_noise)
        elif opt.method == "mle":
            loss = self.compute_loss_mle(event_type, event_time, time_gap, non_pad_mask, num_grid = opt.num_grid)
        elif opt.method == "wsm":
            loss = self.compute_loss_wsm(event_type, event_time, time_gap, non_pad_mask, opt.h_type, opt.CE_coef)
        else:
            raise ValueError("No such method")
        return loss, enc_output
    
    def compute_loss_wsm(self, event_type, event_time, time_gap, non_pad_mask, h_type, alpha = 1):
        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)
        t_var = torch.autograd.Variable(time_gap, requires_grad=True)

        all_intensity, score= self.get_intensity_n_score(t_var, event_type, event_time, time_gap, non_pad_mask)
        score_grad = score_grad = torch.autograd.grad(score.sum(), t_var, create_graph=True)[0][:,1:] * non_pad_mask[:,1:].squeeze(-1)
        # all_intensity, score, score_grad = self.packed_score_manual(t_var, event_type, event_time, time_gap, non_pad_mask)

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
            h = torch.ones_like(event_time[:,1:])
            hprime = torch.zeros_like(event_time[:,1:])
        elif h_type == "two_side_ord":
            h = (max_observed - event_time[:,1:]) * time_gap[:,1:]
            hprime = max_observed - event_time[:,1:] - time_gap[:,1:]
        elif h_type == "one_side_ord":
            h = time_gap[:,1:] ** 2
            hprime = 2 * time_gap[:,1:]       
        elif h_type == "one_side_opt":
            h = time_gap[:,1:]
            hprime = torch.ones_like(time_gap[:,1:])
        else:
            raise ValueError("No such h_type")

    
        WSMLoss = (0.5 * h * score ** 2 + score_grad * h + score * hprime)

        loss = (WSMLoss + alpha * CELoss) * non_pad_mask[:,1:].squeeze(-1)
        return loss
    

    def compute_loss_dsm(self, event_type, event_time, time_gap, non_pad_mask,  var_noise=1, alpha=1, num_noise = 10):
        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)

        noise = var_noise * torch.randn([*time_gap.size(), num_noise], device = time_gap.device)
        t_noise = time_gap[:,:,None] + noise
        t_var = torch.autograd.Variable(t_noise, requires_grad=True)

        cell_states = self.mu(self.enc_output[:,:-1])[:,:,None,:] + (
            # self.eta(self.enc_output[:,:-1])[:,:,None,:]  - self.mu(self.enc_output[:,:-1])[:,:,None,:] ) * torch.exp(
            self.eta(self.enc_output[:,:-1])[:,:,None,:]  - self.mu(self.enc_output[:,:-1])[:,:,None,:] ) * F.softplus(
         -self.gamma(self.enc_output[:,:-1])[:,:,None,:]  * t_var[:,1:,:,None])

        
        all_intensity_noise = self.softplus(cell_states)
        intensity_total = all_intensity_noise.sum(-1)

        intensity_total_log = ((intensity_total+1e-10).log())
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t_var, create_graph=True)[0][:,1:]
        score = intensity_total_grad_t - intensity_total
        noise_score = -noise[:,1:] / var_noise ** 2

        cell_t = self.state_decay(encode_state=self.enc_output[:,:],
                            duration_t=time_gap[:,:,None])
        all_intensity = self.softplus(cell_t)
        sum_intensity  = all_intensity.sum(-1)
        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_time.device)

        type_indices = torch.arange(1, self.num_types + 1, device=event_time.device).view(1, 1, -1)
        type_mask = (event_type.unsqueeze(-1) == type_indices).to(event_time.device)

        type_intensity = (all_intensity * type_mask[:, 1:, :]).sum(-1)
        type_intensity = (all_intensity * type_mask[:,1:,:]).sum(-1)
        
        CELoss = -(type_intensity + 1e-10).log() + (sum_intensity + 1e-10).log()

        loss = (0.5 * (score - noise_score) ** 2 * non_pad_mask[:,1:,:]).sum(-1) / num_noise
        loss += alpha * CELoss * non_pad_mask[:,1:].squeeze(-1)

        return loss

        
        
    def get_intensity_n_score(self, t, event_type, event_time, time_gap, non_pad_mask):
        cell_t = self.state_decay(encode_state=self.enc_output[:,:],
                            duration_t=t[:,:,None])
        all_intensity = self.softplus(cell_t)
        intensity_total = all_intensity.sum(-1)

        intensity_total_log = ((intensity_total+1e-10).log())
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t, create_graph=True)[0][:,1:]
        score = intensity_total_grad_t - intensity_total
        # score_grad = torch.autograd.grad(score.sum(), t, create_graph=True)[0][:,1:] * non_pad_mask[:,1:]

        # return all_intensity, score, score_grad
        return all_intensity, score
        
        
    def compute_loss_mle(self, event_type, event_time, time_gap, non_pad_mask, num_grid=10):
        
        # [batch_size, seq_len-1, hidden_dim]
        cell_t = self.state_decay(encode_state=self.enc_output[:,:],
                            duration_t=time_gap[:, :, None]).squeeze(-2)

        # [batch_size, seq_len-1, num_types]
        lambda_at_event = self.softplus(cell_t)

        # print(lambda_at_event.shape, 'the lambda at event, should be [batch_size, *, num_types]')

        # [batch_size, seq_len, hidden_dim]
        type_mask = 0
        for type_ in range(1, self.num_types+1):
            type_i_mask = (event_type[:,:] == type_).unsqueeze(-1)
            if type_==1:
                type_mask = type_i_mask
            else:
                type_mask = torch.cat((type_mask, type_i_mask), axis=-1)

        assert lambda_at_event.shape[1] == cell_t.shape[1] == event_time.shape[1]-1 == time_gap.shape[1]-1

        # Compute the big lambda integral in equation (8) of NHP paper
        # 1 - take num_mc_sample rand points in each event interval
        # 2 - compute its lambda value for every sample point
        # 3 - take average of these sample points
        # 4 - times the interval length

        # [batch_size, seq - 1, num_sample, event_num]
        sample_dtimes = self.make_dtime_loss_samples(time_gap, num_grid)

        # print(sample_dtimes.shape, 'the sampled duration shape, should be [batch_size, seq_len, num_sample]')

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(encode_state=self.enc_output, sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        # print(lambda_at_event.shape, 'event inten!')
        # print(lambda_t_sample.shape, 'non-event inten!')
        

        non_pad_mask = non_pad_mask.squeeze(-1).bool()

        assert lambda_at_event.shape[1] == lambda_t_sample.shape[1] == time_gap.shape[1]-1 == non_pad_mask.shape[1]-1 == type_mask.shape[1]-1
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_gap[:, 1:],
                                                                        seq_mask=non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # return enc_inten to compute accuracy
        log_likelihood = (event_ll - non_event_ll).sum()

        return -log_likelihood

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

        
        cell_states = self.mu(encode_state[:,:-1, None, :]) + (
            # self.eta(encode_state[:,:-1, None, :]) - self.mu(encode_state[:,:-1, None, :])) * torch.exp(
            self.eta(encode_state[:,:-1, None, :]) - self.mu(encode_state[:,:-1, None, :])) * F.softplus(
         -self.gamma(encode_state[:,:-1, None, :]) * sample_dtimes[:,1:,:,None])

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

        intensity_pred = self.compute_intensity([event_type, event_time,time_gap], opt)
        # intensity_pred = torch.stack(list(intensity_pred.values()), dim=-1).squeeze(2)[:,1:]
        _, type_pred  = torch.max(intensity_pred, dim=-1)

        return type_pred + 1
    
    def compute_score(self, event_type, event_time, time_gap):
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0) 
        non_pad_mask = get_non_pad_mask(event_type)

        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)

        type_embedding = self.layer_type_emb(event_type)

        enc_output = type_embedding + self.layer_position_emb(event_time, time_gap)
        enc_output = self.encoder(
                enc_output,
                enc_output,
                mask=slf_attn_mask
                )
        
        self.enc_output = enc_output

        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)
        t_var = torch.autograd.Variable(time_gap, requires_grad=True)

        cell_t = self.state_decay(encode_state=self.enc_output[:,:],
                            duration_t=t_var[:, :, None])
        all_intensity = self.softplus(cell_t)
        non_pad_mask = non_pad_mask.squeeze(-1)
        intensity_total = all_intensity.sum(-1)*non_pad_mask[:,1:]

        intensity_total_log = ((intensity_total+1e-10).log()*non_pad_mask[:,1:])
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t_var, create_graph=True)[0][:,1:] * non_pad_mask[:,1:]
        score = intensity_total_grad_t - intensity_total
        score_grad = torch.autograd.grad(score.sum(), t_var, retain_graph=True)[0][:,1:]

        return score, score_grad
    

    def score_manual(self, event_type, event_time, time_gap):
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0) 
        non_pad_mask = get_non_pad_mask(event_type)

        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)

        type_embedding = self.layer_type_emb(event_type)

        enc_output = type_embedding + self.layer_position_emb(event_time, time_gap)
        enc_output = self.encoder(
                enc_output,
                enc_output,
                mask=slf_attn_mask
                )
        
        self.enc_output = enc_output

        time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)
        t_var = torch.autograd.Variable(time_gap, requires_grad=True)

        mu_matrix = self.mu(enc_output[:,:-1])
        eta_matrix = self.eta(enc_output[:,:-1])
        gamma_matrix = self.gamma(enc_output[:,:-1])

        self.mu_matrix = mu_matrix
        self.eta_matrix = eta_matrix
        self.gamma_matrix = gamma_matrix
        # non_pad_mask = non_pad_mask.squeeze(-1)

        state_decay = mu_matrix + (eta_matrix - mu_matrix)* torch.exp(-gamma_matrix * t_var[:, 1:, None])
        ds_dt = (eta_matrix - mu_matrix) * ( -gamma_matrix) * torch.exp(-gamma_matrix * t[:, 1:, None])
        d2s_dt2 = (eta_matrix - mu_matrix) * (gamma_matrix**2) * torch.exp(-gamma_matrix * t[:, 1:, None])

        dlambda_dt = ((1 - 1/(1+torch.exp(state_decay))) * ds_dt).sum(-1)
        d2lambda_dt2 = ((1 - 1/(1+torch.exp(state_decay))) * d2s_dt2 + ds_dt**2 * (1/(1+torch.exp(state_decay)) - 1/(1+torch.exp(state_decay))**2)).sum(-1)

        all_intensity = torch.log(1 + torch.exp(state_decay)) * non_pad_mask[:,1:]
        sum_intensity = all_intensity.sum(-1)
        # all_intensity = self.my_softplus(state_decay) * non_pad_mask[:,1:]

        score = 1/(sum_intensity + 1e-10) * dlambda_dt *non_pad_mask[:,1:].squeeze(-1) - sum_intensity *non_pad_mask[:,1:].squeeze(-1)
        score_grad = - 1/(sum_intensity + 1e-10)**2 * dlambda_dt ** 2 * non_pad_mask[:,1:].squeeze(-1) + d2lambda_dt2 / (sum_intensity + 1e-10) * non_pad_mask[:,1:].squeeze(-1) - dlambda_dt * non_pad_mask[:,1:].squeeze(-1)

        return score, score_grad

    def my_softplus(self, x):
        return torch.where(x > 80, x, torch.log(1 + torch.exp(x)))



    def packed_score_manual(self, t, event_type, event_time, time_gap, non_pad_mask):
        
        enc_output = self.enc_output
        mu_matrix = self.mu(enc_output[:,:-1])
        eta_matrix = self.eta(enc_output[:,:-1])
        gamma_matrix = self.gamma(enc_output[:,:-1])
        
        state_decay = self.state_decay(encode_state=self.enc_output[:,:],
                            duration_t=t[:, :, None])

        ds_dt = (eta_matrix - mu_matrix) * ( -gamma_matrix) * torch.exp(-gamma_matrix * t[:, 1:, None])
        d2s_dt2 = (eta_matrix - mu_matrix) * (gamma_matrix**2) * torch.exp(-gamma_matrix * t[:, 1:, None])

        dlambda_dt = ((1 - 1/(1+torch.exp(state_decay))) * ds_dt).sum(-1)
        d2lambda_dt2 = ((1 - 1/(1+torch.exp(state_decay))) * d2s_dt2 + ds_dt**2 * (1/(1+torch.exp(state_decay)) - 1/(1+torch.exp(state_decay))**2)).sum(-1)

        all_intensity = torch.log(1 + torch.exp(state_decay)) * non_pad_mask[:,1:]
        sum_intensity = all_intensity.sum(-1)
        # all_intensity = self.my_softplus(state_decay) * non_pad_mask[:,1:]

        score = 1/(sum_intensity + 1e-10) * dlambda_dt *non_pad_mask[:,1:].squeeze(-1) - sum_intensity *non_pad_mask[:,1:].squeeze(-1)
        score_grad = - 1/(sum_intensity + 1e-10)**2 * dlambda_dt ** 2 * non_pad_mask[:,1:].squeeze(-1) + d2lambda_dt2 / (sum_intensity + 1e-10) * non_pad_mask[:,1:].squeeze(-1) - dlambda_dt * non_pad_mask[:,1:].squeeze(-1)

        return all_intensity, score, score_grad