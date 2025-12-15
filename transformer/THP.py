import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer.Constants as Constants


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    # temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_end_mask(mask):
     # mask: (batch_size, seq_len)
    # This function returns a nonmask label, if 1 represents this is the end of the sequence
    
    device = mask.device
    row_sum = mask.sum(dim=1).squeeze().to(int)  # (batch_size)
    end_mask = torch.zeros(mask.shape[0], mask.shape[1] + 1, 1).to(device)
    end_mask[torch.arange(end_mask.size(0), device=end_mask.device), row_sum , 0] = 1 # (batch_size, seq_len)
    return end_mask


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





class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x




class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn



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
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
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
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
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





class thp_Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=16, d_inner=8,
            n_layers=1, n_head=1, d_k=16, d_v=16, dropout=0.1, opt=None, max_time = None):
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
        self.max_time = max_time
        self.noise_type = opt.noise_type
        self.with_survival = opt.with_survival
        self.alpha_neg = opt.alpha_neg
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
        

        self.Survival = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, 2)
        )

    
    def forward(self, event_type, event_time, time_gap, opt):
        """
        starts from 0, fit t_1,...., t_N   
        """        

        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        self.enc_output = enc_output

        if opt.method == "wsm":
            loss = self.compute_loss_wsm(event_type[:,1:], event_time, time_gap, non_pad_mask, opt.h_type, opt.CE_coef, opt.alpha_survival)
        elif opt.method == "mle":
            loss = self.compute_loss_mle(event_type, event_time, time_gap, num_grid = opt.num_grid)
        elif opt.method == "dsm":
            loss = self.compute_loss_dsm(event_type[:,1:], time_gap, non_pad_mask, opt.noise_var, opt.CE_coef, opt.num_noise, opt.alpha_survival)
        else:
            raise ValueError("No such method")
        return loss, enc_output


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

        survival_loss = ce.mean()
        return survival_loss
    
    def compute_loss_wsm(self, event_type, event_time, time_gap, non_pad_mask, h_type="two_side_op",alpha = 1, alpha_survival = 1.0):
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
        
        
        if h_type == "two_side_op":
            t_prior = event_time[:,:-1]
            h = (max_observed - t_prior)/2 - abs(event_time[:,1:] - (max_observed + t_prior)/2)
            hprime = torch.where(event_time[:,1:] > (max_observed + t_prior)/2, -1, 1)
        elif h_type == "None":
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

        survival_loss = self.get_survival_loss(non_pad_mask, mode = "train", alpha_neg = self.alpha_neg)

    
        WSMLoss = (0.5 * h * score ** 2 + score_grad * h + score * hprime)

        loss = (( WSMLoss + alpha * CELoss) * non_pad_mask[:,1:].squeeze(-1)).sum() + survival_loss * alpha_survival
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

        score = self.get_score(t_var, self.enc_output[:, :-1], non_pad_mask[:, 1:])
        

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
    
    def get_intensity_n_score(self, t, cond, non_pad_mask):
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        self.affect = self.affect_layer(cond)
        self.base = self.base_layer(cond)
        
        intensity = self.intensity_layer(self.affect.unsqueeze(2) * t.unsqueeze(3) + self.base.unsqueeze(2)).squeeze(3) # (batch*len-1*1/num_samples)
        if self.num_types == 1:
            intensity = intensity.unsqueeze(-1)
        all_lambda = intensity * non_pad_mask[:,:, None, :]
        intensity_total = all_lambda.sum(-1)

        intensity_total_log = ((intensity_total+1e-10).log()*non_pad_mask)
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t, create_graph=True)[0]*non_pad_mask
        score = intensity_total_grad_t - intensity_total

        return all_lambda.squeeze(-2), score.squeeze(-1)

    def get_intensity(self, t, cond, non_pad_mask):
        
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        self.affect = self.affect_layer(cond)
        self.base = self.base_layer(cond)
        intensity = self.intensity_layer(self.affect.unsqueeze(2) * t.unsqueeze(3) + self.base.unsqueeze(2)).squeeze(3) 
        if self.num_types == 1:
            intensity = intensity.unsqueeze(-1)
        

        all_lambda = intensity * non_pad_mask[:,:, None, :]
        return all_lambda
            
    

    def get_score(self, t, cond, non_pad_mask):
        # we make sure the shape of t, cond, non_pad_mask are the same (for the first two dimensions)
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        all_intensity = self.get_intensity(t, cond, non_pad_mask) # batch*len*num_samples*num_type
        intensity_total = all_intensity.sum(-1)*non_pad_mask

        intensity_total_log = ((intensity_total+1e-10).log()*non_pad_mask)
        intensity_total_grad_t = torch.autograd.grad(intensity_total_log.sum(), t, create_graph=True)[0]*non_pad_mask
        score = intensity_total_grad_t - intensity_total

        return score


    def compute_loss_mle(self, event_type, event_time, time_gap,  num_grid = 10):

        ll = self.log_likelihood(event_time, time_gap, event_type, num_grid)
        loss = -ll.sum()
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
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        self.enc_output = enc_output
        intensity_pred = self.get_intensity(time_gap, self.enc_output[:, :-1], non_pad_mask[:, 1:]).squeeze(2)
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

    # def get_intensity(self, t, cond)

    def eval_intensity(self, event_type, event_time, time_gap, num_grid=30):

        #evaluate intensity function
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
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
        




    def compute_event(self, time_gap, event_type, non_pad_mask):
        # time_gap : batch*seq_len-1
        # event_type : batch*seq_len-1
        # non_pad_mask : batch*seq_len-1


        type_mask = torch.zeros([*event_type.size(), self.num_types], device=event_type.device)
        for i in range(self.num_types):
            type_mask[:, :, i] = (event_type == i + 1).bool().to(event_type.device)

        all_lambda = self.get_intensity(time_gap, self.enc_output[:, :-1], non_pad_mask).squeeze(2)

        event = torch.sum(all_lambda * type_mask, dim=2)
        event += math.pow(10, -9)
        event.masked_fill_(~non_pad_mask.squeeze(2).bool(), 1.0)
        result = torch.log(event+1e-10) * non_pad_mask.squeeze(2)
        return result

    def compute_integral_unbiased(self, time_gap, cond, non_pad_mask, num_grid):
        """ Log-likelihood of non-events, using Monte Carlo integration. """


        num_samples = num_grid
        if self.normalize == 'log':
            time_low = min(-1.0,time_gap.min()-1.0)
        else:
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


    def log_likelihood(self, event_time, time_gap, event_type, num_grid):
        """ Log-likelihood of sequence. """

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
        










