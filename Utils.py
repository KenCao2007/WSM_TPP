import torch
from transformer import Constants
from transformer.THP import get_non_pad_mask
from tqdm import tqdm




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