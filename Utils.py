import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from transformer import Constants
import matplotlib.pyplot as plt
from transformer.Models import get_non_pad_mask
from tqdm import tqdm
import pickle

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    # temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))



def compute_event(model, event_time, time_gap, event_type, non_pad_mask):
    
    type_mask = torch.zeros([*event_type.size(), model.num_types], device=event_time.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (event_type == i + 1).bool().to(event_time.device)

    all_lambda = model.get_intensity(time_gap, event_type, event_time, time_gap, non_pad_mask).squeeze(2)

    event = torch.sum(all_lambda * type_mask[:, 1:, :], dim=2)
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask[:,1:].squeeze(2).bool(), 1.0)
    result = torch.log(event+1e-10) * non_pad_mask[:,1:].squeeze(2)
    return result

def compute_integral_unbiased(model, event_time, time_gap, event_type, non_pad_mask, num_grid):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = num_grid
    if model.normalize == 'log':
        time_low = min(-1.0,time_gap.min()-1.0)
    else:
        time_low = 0
    temp_time = (time_gap.unsqueeze(2) - time_low) * \
                torch.rand([*time_gap.size(), num_samples], device=event_time.device) + time_low
    
    if model.num_types >= 100:
        all_lambda = None
        for i in range(num_samples):
            lambda_i = model.get_intensity(temp_time[:,:,i:i+1], event_type, event_time, time_gap, non_pad_mask)
            if all_lambda == None:
                all_lambda = torch.sum(lambda_i, dim=(2,3)) 
            else:
                all_lambda += torch.sum(lambda_i, dim=(2,3)) 
        all_lambda /= num_samples
    else:
        all_lambda = model.get_intensity(temp_time, event_type, event_time, time_gap, non_pad_mask)
        all_lambda = torch.sum(all_lambda, dim=(2,3)) / num_samples

    unbiased_integral = all_lambda * (time_gap - time_low) * non_pad_mask.squeeze(-1)[:,1:]
    return unbiased_integral


def log_likelihood(model, event_time, time_gap, event_type, num_grid):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(event_type)

    # event log-likelihood
    event_ll = compute_event(model, event_time, time_gap, event_type, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    non_event_ll = compute_integral_unbiased(model, event_time, time_gap, event_type, non_pad_mask, num_grid)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self,  tgt_vocab_size, ignore_index=-100):
        label_smoothing = 0.0
        assert 0.0 <= label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target, logit):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes
        if logit:
            log_prb = F.log_softmax(output, dim=-1)
        else:
            log_prb = (output+1e-10).log()

        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss




def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se





def hawkes_next_time_prediction(history):
    
    mu = 1
    alpha = 1
    beta = 2

    N = 1e4
    grid_ = 1e-2
    
    t_n = history[-1]

    diff_tn = torch.tensor(t_n).unsqueeze(-1) - history[:-1]


    Lambda_tn = mu * t_n - torch.sum(alpha/beta * torch.exp(-beta * diff_tn), dim=-1)
    Rimen = 0
    for i in np.arange(1,N):
        s = t_n + i * grid_

        diff_ = torch.tensor(s).unsqueeze(-1) - history

        intensity_s = mu + torch.sum((alpha * torch.exp(-beta * diff_)), dim=-1)
        Lambda_s = mu * s - torch.sum((alpha/beta * torch.exp(-beta * diff_)), dim=-1)
        
        Rimen += s*intensity_s/torch.exp(Lambda_s) * grid_
    
    pred_next_t = torch.exp(Lambda_tn) * Rimen
    return pred_next_t


def langevin_sampling(model, history):
        sampling_nums = 100
        sampling_step = 1
        n = 0
        t_n = torch.tensor(history[-1]).float().to('cuda')
        while n < sampling_nums:
            noise = torch.randn(1).squeeze(-1).to('cuda')
            score_ = model.get_score(t_n, history)

            tmp = score_ * sampling_step/2 + sampling_step**0.5 * noise
            if tmp < 0:
                continue
            else:
                n += 1
                t_n += tmp


        return t_n


def synthetic_gt_comparison(model, testloader, opt):

    data_iter = iter(testloader)
    batch = next(data_iter)
    
    #We can debiase it, but we do not have to
    event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

    non_pad_mask = event_time > 0
    #### learned intensity
    num_types = opt.num_types
    if  not opt.debug:
        _, enc_dict = model(event_type, event_time, time_gap, opt)
    else:
        _, enc_dict, _ = model(event_type, event_time, time_gap, opt)
    intensity_learned = {}

    if opt.model == "thp":
        intensity = model.get_intensity(time_gap[:,1:], event_type, event_time, time_gap, non_pad_mask.unsqueeze(-1))
        for i in range(num_types):
            intensity_learned[i] = intensity[:,:,:,i].squeeze(-1)
    else:
        for i in range(num_types):
            intensity_learned[i] = (model.intensity_decoder[i](enc_dict[i+1]) * non_pad_mask.unsqueeze(-1))[:,1:]
    
    #### ground truth intensity 
    event_time = event_time.cpu().numpy()
    event_type = event_type.cpu().numpy()

    bs, _ = event_time.shape



    def uni_hawkes_inten(grids, events, parameters):

        event_time, event_type =events
        mu, alpha, beta = parameters
        intensity = []
        for grid_ in grids:
            gap = grid_ - event_time
            gap = gap[gap>0]
            intensity.append(mu + np.sum(alpha * np.exp(-beta * gap)))

        return {0: intensity}

    def half_sin_inten(grids, events, parameters):
        '''
        input seqs: sequence_length
        parameters: mu, alpha
        '''
        event_time, event_type = events
        mu, alpha = parameters

        num_types = len(mu)
        grids_len = len(grids)
        intensity = {}
        for m in range(num_types):
            intensity[m] = []
            for i in range(grids_len):
                
                current_t = grids[i]
                history_ = event_time[event_time<current_t]
                intensity_i = mu[m]
                if not len(history_):
                    intensity[m].append(intensity_i)
                    continue

                for j in range(len(history_)):
                    gap_ = current_t - history_[j]
                    j_type = event_type[j]-1 # type starting from 1
                    if gap_ <= np.pi:
                        effect = alpha[m][j_type] * np.sin(gap_)
                    else:
                        effect = 0
                    intensity_i += effect
                intensity[m].append(intensity_i)
        return intensity
    
    def exp_decay_inten(grids, events, parameters):
        '''
        input seqs: sequence_length
        parameters: mu, alpha
        '''
        event_time, event_type = events
        mu, alpha = parameters

        num_types = len(mu)
        grids_len = len(grids)
        intensity = {}
        for m in range(num_types):
            intensity[m] = []
            for i in range(grids_len):
                
                current_t = grids[i]
                history_ = event_time[event_time<current_t]
                intensity_i = mu[m]
                if not len(history_):
                    intensity[m].append(intensity_i)
                    continue

                for j in range(len(history_)):
                    gap_ = current_t - history_[j]
                    j_type = event_type[j]-1 # type starting from 1
                    effect = alpha[m][j_type] * np.exp(-5 * gap_)
                    intensity_i += effect
                intensity[m].append(intensity_i)
        return intensity


    if 'half-sin' in opt.data:
        mu=[0.2,0.2]
        alpha=[[0.33,0.1],[0.05,0.33]]
        parameters=[mu,alpha]
        inten_func = half_sin_inten
        T = 200
    elif 'hawkes' in opt.data:
        mu = 1
        alpha = 1
        beta = 2
        parameters = [mu, alpha, beta]
        inten_func = uni_hawkes_inten
        T = 20
    elif 'exp-decay' in opt.data:
        mu=[1.0,1.0]
        alpha=[[1.6,0.2],[1.0,1.0]]
        parameters = [mu, alpha]
        inten_func = exp_decay_inten
        T = 10
    for seq_id in range(bs):
        time_ = event_time[seq_id,]
        type_ = event_type[seq_id,]

        mask = time_>0

        time_ = time_[mask]
        type_ = type_[mask]
        

        grids=np.linspace(0,T,2000)    # intensity function curve
        intensity_gt = inten_func(grids, [time_, type_], parameters)

        plt.figure(1,figsize=(10,5)) 
        for m in range(num_types):
            intensity_learned_m_ = (intensity_learned[m].squeeze(-1).detach().cpu().numpy())[seq_id,:]
            intensity_learned_m_ = intensity_learned_m_[mask[1:]]
            plt.subplot(num_types,1,m+1)
            plt.plot(grids,intensity_gt[m],'r-',label=('hawkes %s intensity'%(m+1)))
            # plt.scatter(x=time_,y=intensity_learned_m_,label=('learned %s intensity'%(m+1)))
            plt.plot(time_[1:],intensity_learned_m_,label=('learned %s intensity'%(m+1)))
            plt.ylim(0)
            plt.legend(loc='best',frameon=0)
        plt.savefig(opt.results_saved_path + '/synthetic ground truth vs learned intensity.png')
        plt.close()
        return



    # plt.figure(1,figsize=(7,5))
    # plt.subplot(1,1,1)             # points position
    # for m in range(M):
    #     plt.plot([0,T],[m,m],'r-',lw=1,alpha=0.6)
    #     plt.plot(points_hawkes[m],[m]*len(points_hawkes[m]),linestyle='None', marker='|', markersize=10,label=('hawkes %s'%(m+1)))
    # plt.title('multivariate hawkes process')
    # plt.ylim(-1,2)
    # plt.legend(loc='best',frameon=0)
    # plt.savefig('hawkes points.png')

def grids_events(event_time, event_type, opt):

        # Step 1: Left shifted.
        # find the min event time and max event time. Note that padding entry 0 is excluded
        event_time_clone = event_time.clone()
        sorted_, _ = torch.sort(torch.unique(event_time_clone))
        
        min_ = sorted_[1] if sorted_[0] == Constants.PAD else sorted_[0]
        max_ = sorted_[-1]
        batch_, _ = event_time_clone.shape

        num_grids = 50
        grid_length = (max_ - min_) / num_grids
        # Manually construct grids.
        # grids = torch.arange(min_, max_, Constants.GRID_LENGTH).to(opt.device) # generate grids with step=grid_length (0.1 as default) from min event time to max event time.
        grids = torch.arange(min_, max_, grid_length).to(opt.device)
        grids = grids.repeat(event_time_clone.shape[0],1) # stacking grids batch_size times.
        
        grids_type = torch.ones_like(grids) * Constants.GRID  # Set the "Type" to be -1 at grids

        for unique_t in torch.unique(event_time_clone)[1:]: # the first element of torch.unique(event_time) is padding 0, exclude it.
            grids[grids==unique_t] = unique_t + 1e-3

        
        mix_time = torch.cat((event_time_clone, grids), dim=1)
        mix_type = torch.cat((event_type, grids_type), dim=1)

        # set padding entries to be a very large value,
        # ensuring their positions won't change during sorting the combination of Non-event grids and Event time
        mix_time[mix_time == 0.0] = 1e10

        sorted_time, indice = mix_time.sort()
        sorted_types = torch.zeros_like(sorted_time)

        for i in range(batch_):
            sorted_types[i, :] = mix_type[i][indice[i]]
        sorted_types = sorted_types.long()
        sorted_time[sorted_time >= 1e10] = 0  # set padding entry back to be 0

        sorted_time.to(opt.device)
        sorted_types.to(opt.device)

        return sorted_time, sorted_types, grid_length


def thp_loglikelihood(model, dataloader, opt):

    total_tll = 0
    total_num_events = 0
    model.eval()
    method = opt.method
    num_grid = opt.num_grid
    for idx, batch in enumerate(tqdm(dataloader, mininterval=2, desc='  - (Evaluating) ', leave=False)):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        if event_time.shape[1] == 0:
            break
        event_time, time_gap, event_type = data_transformation(event_time, time_gap, event_type, opt)
        non_pad_mask = get_non_pad_mask(event_type)
        event_time = event_time * non_pad_mask.squeeze(-1)

        opt.method = "mle"
        opt.num_grid = 100
        loss, _ = model(event_type, event_time, time_gap, opt)
        # model(event_type, event_time, time_gap, opt)
        # loss = model.compute_loss_mle(event_type, event_time, time_gap, non_pad_mask)
        total_tll += -loss
        num_events = event_type.ne(Constants.PAD).sum().item()
        total_num_events += num_events
    opt.method = method
    opt.num_grid = num_grid
    model.train()
    return total_tll/total_num_events
    
def sahp_loglikelihood(model, dataloader, opt):

    total_tll = 0
    total_num_events = 0
    for idx, batch in enumerate(tqdm(dataloader, mininterval=2, desc='  - (Evaluating) ', leave=False)):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        if event_time.shape[1] == 0:
            break
        event_time = (event_time - event_time[:, 0:1])

        event_time = event_time[:, 1:]
        event_type = event_type[:, 1:]
        time_gap = time_gap[:, 1:]
        if opt.data_name in ['retweet', 'taobao','stackoverflow']:
            event_time /= 10
        if opt.data_name in ['earthquake']:
            event_time  *= 10

        non_pad_mask = get_non_pad_mask(event_type)
        event_time = event_time * non_pad_mask.squeeze(-1)

        loss, _ = model(event_type, event_time, time_gap, opt)
        
        num_events = event_type.ne(Constants.PAD).sum().item()
        total_num_events += num_events

        total_tll += loss*num_events
    return total_tll/total_num_events

def evahp_score_loglikelihood(model, dataloader, opt):
    opt.train_able=False
    data_iter = iter(dataloader)
    batch = next(data_iter)
    event_time, _, event_type = map(lambda x: x.to(opt.device), batch)
    non_pad_mask = get_non_pad_mask(event_type)
    time_gap = torch.cat((torch.zeros_like(event_time[:,0:1]), (event_time[:,1:] - event_time[:,:-1])*non_pad_mask[:,1:].squeeze(-1)),axis=-1)
    type_mask = 0
    for type_ in range(1, opt.num_types+1):
        type_i_mask = (event_type[:,:] == type_).unsqueeze(-1)
        if type_==1:
            type_mask = type_i_mask
        else:
            type_mask = torch.cat((type_mask, type_i_mask), axis=-1)
    # print(event_type.shape, event_time.shape, time_gap.shape, non_pad_mask.shape, type_mask.shape)

    ll, num_events = model.loglike_loss([event_type, event_time, time_gap, non_pad_mask, type_mask], opt)

    return ll.item()/num_events

def thp_score_fine_grained_plotting(model, dataloader, opt):
    batch = next(iter(dataloader))
    # batch = next(data_iter)
    event_time, _, event_type = map(lambda x: x.to(opt.device), batch)
    intensity_fg, mixed_time = model.compute_fine_grained_intensity(event_type, event_time, opt)

    plt.figure()
    plt.plot(mixed_time[0,1:].cpu().detach().numpy(), intensity_fg[1][0, :])
    plt.savefig('zaijian.png')

    with open('results/{}_{}_intensity_fg.pkl'.format(opt.data_name, opt.model), 'wb') as f:
        pickle.dump([intensity_fg, mixed_time, event_time], f)
    

def eval_accuracy(model, dataloader, opt):
    # batch = next(iter(dataloader))
    
    total_num_correct = 0
    total_num_pred = 0
    model.eval()
    for idx, batch in enumerate(tqdm(dataloader, mininterval=2, desc='  - (Evaluating) ', leave=False)):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        if event_time.shape[1] == 0:
            break
        event_time, time_gap, event_type = data_transformation(event_time, time_gap, event_type, opt)

        non_pad_mask = get_non_pad_mask(event_type).squeeze(-1)
        event_time = event_time * non_pad_mask
        opt.method = "wsm"
        type_pred = model.predict(event_type, event_time, time_gap, opt) * non_pad_mask
        total_num_correct += torch.sum(type_pred == event_type).item() - torch.sum(event_type== 0).item()
        total_num_pred += torch.sum(non_pad_mask).item()
    return total_num_correct/total_num_pred
    

def data_transformation(event_time, time_gap, event_type, opt):
    event_time = event_time.type(torch.float64)
    time_gap = time_gap.type(torch.float64)
    if opt.data_name != "exp-decay-multivariate" and opt.data_name != "half-sin_multivariate":
        event_time = (event_time - event_time[:, 0:1])[:,1:]
        event_type = event_type[:,1:]
        time_gap = time_gap[:,2:]
    else:
        time_gap = time_gap[:,1:]
    # if opt.data_name == "earthquake":
    #     event_time *= 10
    #     time_gap *= 10
    if opt.data_name in ['retweet']:
        event_time /= 100
        time_gap /= 100
    if opt.data_name in ['taobao']:
        event_time *= 5
        time_gap *= 5
        # if opt.data_name not in ["exp-decay-multivariate", "earthquake", "half-sin_multivariate","taobao","retweet"]:
        #     event_time /= 5
        #     time_gap /= 5
        # if opt.data_name == "taobao":
        #     event_time /= 500
        #     time_gap /= 500
        # if opt.data_name == 'half-sin_multivariate':
        #     event_time /= 2
        #     time_gap /= 2
    if opt.seq_trunc and opt.train_able:
        min_length = (event_type != 0).sum(dim = 1).min().item()
        event_type[:, min_length:] = 0
        event_time[:, min_length:] = 0
        time_gap[:, min_length:] = 0
        # opt.h_type = "one_side"
    if opt.delete_outlier and opt.train_able:
        max_observed =  torch.max(event_time, axis = 1)[0]
        median = max_observed.median()
        std = max_observed.std()
        time_gap = time_gap[torch.abs(max_observed - median) < 2.5*std,:]
        event_type = event_type[torch.abs(max_observed - median) < 2.5*std,:]
        event_time = event_time[torch.abs(max_observed - median) < 2.5*std,:]
    return event_time, time_gap, event_type