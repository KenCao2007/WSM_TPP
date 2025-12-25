import torch
from transformer import Constants
from transformer.THP import get_non_pad_mask
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def conditional_logistic_intensity(t, m, history, M, parameters):
    """
    History-dependent logistic intensity with first-event constant intensity.

    return:
        float, λ_m(t | history)
    """
    beta_param = parameters[0]
    c_param = parameters[1]
    c0_param = None
    if len(parameters) >= 3:
        c0_param = parameters[2]

    # allow beta, c, c0 to be scalar or vector by dimension
    if np.ndim(beta_param) == 0:
        beta_m = float(beta_param)
    else:
        beta_m = float(beta_param[m])

    if np.ndim(c_param) == 0:
        c_m = float(c_param)
    else:
        c_m = float(c_param[m])

    if c0_param is None:
        # if c0 is not given, use c_m by default
        c0_m = c_m
    else:
        if np.ndim(c0_param) == 0:
            c0_m = float(c0_param)
        else:
            c0_m = float(c0_param[m])

    # find the last event time T_last before t for all types
    last_event_time = 0.0
    has_past = False

    for n in range(M):
        hist_n = history[n]  
        for ti in reversed(hist_n):
            if ti < t:
                if (not has_past) or (ti > last_event_time):
                    last_event_time = ti
                    has_past = True
                break  

    if not has_past:
        lam = c0_m
    else:
        tau = t - last_event_time
        lam = c_m / (1.0 + beta_m * np.exp(c_m * tau))

    return lam


def intensity_exp_decay(t,m,history,M,parameters):   #left continue    beta_12 is influence from 2 to 1
    mu=parameters[0][m]   #value
    alpha=parameters[1][m]   #vector
    intensity=0
    for n in range(M):
        for i in range(len(history[n])):
            if history[n][i]>=t:
                break
            else:
                intensity+=alpha[n]*np.exp(-5.* (t-history[n][i]))
    return mu+intensity

def intensity_half_sin(t,m,history,M,parameters):   #left continue    beta_12 is influence from 2 to 1
    mu=parameters[0][m]   #value
    alpha=parameters[1][m]   #vector
    intensity=0
    for n in range(M):
        for i in range(len(history[n])):
            if history[n][i]>=t:
                break
            elif t-history[n][i]<np.pi:
                intensity+=alpha[n]*np.sin((t-history[n][i]))
    return mu+intensity

def intensity_poisson(t,m,history,M,parameters):   #left continue    beta_12 is influence from 2 to 1
    mu=parameters[0][m]   #value
    return mu

def seq_to_points_hawkes(time_seq, type_seq, M):

    # get points_hawkes
    points_hawkes = []
    for m in range(M):
        points_hawkes.append([time_seq[i].item() for i in range(len(time_seq)) if type_seq[i].item()-1 == m])
    return points_hawkes

def eval_intensity(model, dataloader, opt, tll=None):

    model.eval()
    for idx, batch in enumerate(tqdm(dataloader, mininterval=2, desc='  - (Evaluating) ', leave=False)):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        event_time, time_gap, event_type = data_transformation(event_time, time_gap, event_type, opt)
        non_pad_mask = get_non_pad_mask(event_type)
        event_time = event_time * non_pad_mask.squeeze(-1)

        eval_intensity, eval_time = model.eval_intensity(event_type, event_time, time_gap)

        seq_len = non_pad_mask[:, 1:, 0].sum(dim=1)
        if opt.data_name == "half-sin_multivariate":
            mu=[1.,1.5]
            alpha=[[0.33,0.1],[0.05,0.33]]
            parameters=[mu,alpha]
        elif opt.data_name == "exp_decay":
            mu=[2.0,2.5]
            alpha=[[0.33,0.1],[0.05,0.33]]
            parameters=[mu,alpha]
        elif opt.data_name == "conditional_logistic":
            beta=0.02
            c=2.0
            c0=2.0
            parameters=[beta,c,c0]
        for b in range(event_type.shape[0]):
            plt.clf()
            plt.figure()
            if tll is not None:
                plt.title(f'Seq {b} Intensity, TLL: {tll.item():.2f}')
            else:
                plt.title(f'Seq {b} Intensity')
            time_stamp = event_time[b, 1:seq_len[b].long()+1]
            points_hawkes = seq_to_points_hawkes(time_stamp, event_type[b, 1:seq_len[b].long()+1], opt.num_types)

            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            for k in range(opt.num_types):
                type_intensity = eval_intensity[b, :, :, k].flatten().detach().cpu().numpy()
                type_time = eval_time[b, :, :].flatten().detach().cpu().numpy()

                idx = np.argsort(type_time)
                type_time = type_time[idx]
                type_intensity = type_intensity[idx]

                # for better visualization, we delete intensity values that are zero
                mask = type_intensity > 1e-5
                type_time = type_time[mask]
                type_intensity = type_intensity[mask]

                if opt.num_types == 1:
                    true_color = colors[0]
                    pred_color = colors[1] if len(colors) > 1 else colors[0]
                else:
                    base_color = colors[k % len(colors)]
                    true_color = base_color
                    pred_color = base_color
                # ==========================================

                if opt.data_name in ["half-sin_multivariate", "exp_decay", "conditional_logistic"]:

                    x = np.linspace(0, opt.max_event_time, 2000)    # intensity function curve
                    if opt.data_name == "half-sin_multivariate":
                        y = np.fromiter([intensity_half_sin(xi, k, points_hawkes, opt.num_types, parameters) for xi in x], np.float64)
                    elif opt.data_name == "exp_decay":
                        y = np.fromiter([intensity_exp_decay(xi, k, points_hawkes, opt.num_types, parameters) for xi in x], np.float64)
                    elif opt.data_name == "conditional_logistic":
                        y = np.fromiter([conditional_logistic_intensity(xi, k, points_hawkes, opt.num_types, parameters) for xi in x], np.float64)

                    plt.plot(x, y, linestyle='--', color=true_color, label=f'Type {k} True Intensity')

                plt.plot(type_time, type_intensity, color=pred_color, label=f'Type {k} Predicted Intensity')

                plt.plot(points_hawkes[k], [k]*len(points_hawkes[k]),
                        linestyle='None', marker='|', markersize=10,
                        label=('hawkes %s' % (k+1)))


            plt.scatter(time_stamp.cpu(), -1. * torch.ones_like(time_stamp).cpu(), color='red', marker='x', label='Event Time')
            plt.xlabel('Time')
            plt.ylabel('Intensity')
            plt.legend()
            plt.savefig(opt.results_saved_path + f'/intensity_seq{b}.pdf')
            plt.close()


        break
    return



def eval_loglikelihood(model, dataloader, opt):

    total_tll = 0
    total_num_events = 0
    model.eval()
    method = opt.method
    num_grid = opt.num_grid
    for idx, batch in enumerate(tqdm(dataloader, mininterval=2, desc='  - (Evaluating) ', leave=False)):
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        if event_time.shape[1] == 0:
            break
        if event_time.dim() == 1:
            event_time = event_time.unsqueeze(0)
            time_gap = time_gap.unsqueeze(0)
            event_type = event_type.unsqueeze(0)
        event_time, time_gap, event_type = data_transformation(event_time, time_gap, event_type, opt)
        non_pad_mask = get_non_pad_mask(event_type)
        event_time = event_time * non_pad_mask.squeeze(-1)

        opt.method = "mle"
        opt.num_grid = 100

        # this is for debugging
        # model.eval_intensity(event_type, event_time, time_gap)

        #######################
        loss, _ = model(event_type, event_time, time_gap, opt)
        # loss /= non_pad_mask.sum().item()
        # model(event_type, event_time, time_gap, opt)
        # loss = model.compute_loss_mle(event_type, event_time, time_gap, non_pad_mask)
        total_tll += -loss
        # num_events = event_type.ne(Constants.PAD).sum().item()
        num_events = non_pad_mask.sum().item()
        total_num_events += num_events
    opt.method = method
    opt.num_grid = num_grid
    model.train()
    return total_tll/total_num_events
        

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
        type_pred = model.predict(event_type, event_time, time_gap, opt) * non_pad_mask[:, 1:]
        total_num_correct += torch.sum(type_pred == event_type[:, 1:]).item() - torch.sum(event_type[:, 1:] == 0).item()
        total_num_pred += torch.sum(non_pad_mask[:, 1:]).item()
    return total_num_correct/total_num_pred
    

def data_transformation(event_time, time_gap, event_type, opt):
    event_time = event_time.type(torch.float64)
    time_gap = time_gap.type(torch.float64)
    time_gap = time_gap[:, 1:]
    return event_time, time_gap, event_type