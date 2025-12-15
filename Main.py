import argparse
import numpy as np
import logging
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from Utils import eval_loglikelihood
from preprocess.Dataset import get_dataloader
from transformer.THP import thp_Transformer, get_non_pad_mask
from tqdm import tqdm
from transformer.SAHP import SAHP
import Utils
import os
eval_after = 10
eval_gap = 10



def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    logging.info('Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    logging.info('Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    logging.info('Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')


    min_event_time = min([ item['time_since_start'] for data in train_data + dev_data +test_data for item in data])
    max_event_time = max([ item['time_since_start'] for data in train_data + dev_data +test_data for item in data])

    if opt.data_name in ['half-sin_multivariate', 'exp_decay','conditional_logistic']:
        min_event_time = 0.0
        max_event_time = 1.0

    # normalize the event time
    if opt.normalize_time:
        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                train_data[i][j]['time_since_start'] = (train_data[i][j]['time_since_start'] - min_event_time) / (max_event_time - min_event_time) * opt.normalize_scale
                train_data[i][j]['time_since_last_event']  = (train_data[i][j]['time_since_last_event']) / (max_event_time - min_event_time) * opt.normalize_scale
            train_data[i][0]['time_since_last_event'] = train_data[i][0]['time_since_start']  # set the first event's time_since_last_event
            if opt.data_name not in ['taxi']:
                train_data[i].insert(0, {'time_since_start': 0.0, 'time_since_last_event': 0.0, 'type_event': 0})
        for i in range(len(dev_data)):
            for j in range(len(dev_data[i])):
                dev_data[i][j]['time_since_start'] = (dev_data[i][j]['time_since_start'] - min_event_time) / (max_event_time - min_event_time) * opt.normalize_scale
                dev_data[i][j]['time_since_last_event'] = (dev_data[i][j]['time_since_last_event']) / (max_event_time - min_event_time) * opt.normalize_scale
            dev_data[i][0]['time_since_last_event'] = dev_data[i][0]['time_since_start']  # set the first event's time_since_last_event
            if opt.data_name not in ['taxi']:
                dev_data[i].insert(0, {'time_since_start': 0.0, 'time_since_last_event': 0.0, 'type_event': 0})
        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                test_data[i][j]['time_since_start'] = (test_data[i][j]['time_since_start'] - min_event_time) / (max_event_time - min_event_time) * opt.normalize_scale
                test_data[i][j]['time_since_last_event'] = (test_data[i][j]['time_since_last_event']) / (max_event_time - min_event_time) * opt.normalize_scale
            test_data[i][0]['time_since_last_event'] = test_data[i][0]['time_since_start']  # set the first event's time_since_last_event
            if opt.data_name not in ['taxi']:
                test_data[i].insert(0, {'time_since_start': 0.0, 'time_since_last_event': 0.0, 'type_event': 0})

    
    if opt.data_name in ['retweet','taobao','stackoverflow','taxi']:
        # for each list in data[i], we minus data[i][0]['time_since_start'] for each time, and drop the first element
        for i in range(len(train_data)):
            for j in range(1,len(train_data[i])):
                train_data[i][j]['time_since_start'] -= train_data[i][0]['time_since_start']
            train_data[i][0]['time_since_start'] = 0.
        for i in range(len(dev_data)):
            for j in range(1,len(dev_data[i])):
                dev_data[i][j]['time_since_start'] -= dev_data[i][0]['time_since_start']
            dev_data[i][0]['time_since_start'] = 0.
        for i in range(len(test_data)):
            for j in range(1,len(test_data[i])):
                test_data[i][j]['time_since_start'] -= test_data[i][0]['time_since_start']
            test_data[i][0]['time_since_start'] = 0.
            
    elif opt.data_name in ['half-sin_multivariate', 'exp_decay','conditional_logistic']:
        for i in range(len(train_data)):
            # insert 0 at the beginning of the list
            train_data[i].insert(0, {'time_since_start': 0.0, 'time_since_last_event': 0.0, 'type_event': 1})
        for i in range(len(dev_data)):
            dev_data[i].insert(0, {'time_since_start': 0.0, 'time_since_last_event': 0.0, 'type_event': 0})
        for i in range(len(test_data)):
            test_data[i].insert(0, {'time_since_start': 0.0, 'time_since_last_event': 0.0, 'type_event': 0})

    # get maximum event time in all data
    max_event_time = max([ item['time_since_start'] for data in train_data + dev_data +test_data for item in data])
    logging.info('Maximum event time: {}'.format(max_event_time))
    opt.max_event_time = max_event_time
    
    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types, max_event_time


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """


    model.train()
    
    total_loss = 0
    total_preds = 0
    idx = 0
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        idx += 1
        # logging.info('batch: {}'.format(idx))
        """ prepare data """

        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        event_time, time_gap, event_type = Utils.data_transformation(event_time, time_gap, event_type, opt)
            
        non_pad_mask = get_non_pad_mask(event_type)
        event_time = event_time * non_pad_mask.squeeze(-1)
        """ forward """
        optimizer.zero_grad()
        num_preds = torch.tensor(non_pad_mask.shape[0]) * 1.0

        loss_, _ = model(event_type, event_time, time_gap, opt)
        loss = loss_.sum()
        total_loss+=loss.item()
        total_preds+=torch.sum(num_preds)

        loss.backward()
        """ update parameters """
        optimizer.step()
    return total_loss, total_preds



def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """
    opt.train_able=False
    best_tll = eval_loglikelihood(model, validation_data, opt)
    logging.info('Initial testing LogLikelihood: {}'.format(best_tll))
    torch.save(model.state_dict(), opt.results_saved_path + opt.model_saved_name + '.pth')
    logging.info("saving models to {}".format(opt.results_saved_path + opt.model_saved_name + '.pth'))
    start_time = time.time()
    opt.train_able=True
    tr_loss = []
    ll = {'train': [], 'test': []}
    # former = model.mu
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        logging.info('[ Epoch {}]'.format(epoch))
    
        start = time.time()
        total_loss, total_preds = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        logging.info('  - (Training)    objective function: {ll: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=total_loss/total_preds, elapse=(time.time() - start) / 60))

        tr_loss.append(total_loss/total_preds.cpu().numpy())

        """ evaluate the model """
        if epoch_i >= eval_after and epoch_i % eval_gap == 0:
            ll_func = eval_loglikelihood
            opt.train_able=False
            with torch.no_grad():  
                test_ll = ll_func(model, validation_data, opt)
            ll['test'].append(test_ll)
            logging.info('The testing LogLikelihood is {}'.format(test_ll))
            opt.train_able=True
            if test_ll > best_tll:
                best_tll = test_ll
                logging.info('New best testing LogLikelihood: {}'.format(best_tll))
                torch.save(model.state_dict(), opt.results_saved_path + opt.model_saved_name + '.pth')
                logging.info("saving models to {}".format(opt.results_saved_path + opt.model_saved_name + '.pth'))
    
    if epoch_i < eval_after:
        torch.save(model.state_dict(), opt.results_saved_path + opt.model_saved_name + '.pth')
        logging.info("saving models to {}".format(opt.results_saved_path + opt.model_saved_name + '.pth'))


    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {total_time:3.3f} min'.format(total_time=total_time/60))
        
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(tr_loss, label='wsm objective function')
    plt.savefig(opt.results_saved_path + '/loss_curve.png')
    plt.close()
    with open(opt.results_saved_path + '/train_loss.pkl', 'wb') as f:
        pickle.dump(tr_loss, f)
            





def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, default="conditional_logistic", choices=[ "half-sin_multivariate","stackoverflow", "retweet", "taobao", "taxi", "exp_decay","conditional_logistic"])

    parser.add_argument('-epoch', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_model', type=int, default=16)
    parser.add_argument('-d_inner_hid', type=int, default=8)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=2)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-3)

    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-mode', type=str, default='intensity') # score matching mode: build by intensity or model score directly
    parser.add_argument('-method', type=str, choices=["wsm","mle","dsm"], default='wsm')
    parser.add_argument('-model', type=str, choices=['thp_transformer', "sahp"], default='thp_transformer')
    parser.add_argument('-train_able', type=int, default=1)
    parser.add_argument('-load_model', type=int, default=0)
    parser.add_argument('-num_grid', type=int, default=5)
    parser.add_argument('-CE_coef', type=float, default=10.0, help="weight for label cross entropy loss")
    parser.add_argument('-alpha_survival', type = float, default=50.0, help="weight for survival loss")
    parser.add_argument('-alpha_neg', type=float, default=50., help='negative class weight for survival loss')
    parser.add_argument('-h_type', type=str, default="two_side_op", choices=['None','two_side_op','two_side_ord','one_side_ord','one_side_opt'])
    parser.add_argument('-noise_var', type=float, default=0.5)
    parser.add_argument('-num_noise', type=int, default=50)
    parser.add_argument('-seed', type=int, default=4)
    parser.add_argument('-normalize_time', type=int, default=0)
    parser.add_argument('-normalize_scale', type=float, default=1.0, help="scale for time normalization")
    parser.add_argument('-noise_type', type=str, default='lognormal', choices=['normal', 'lognormal'])
    parser.add_argument('-with_survival', type=int, default=1)
    parser.add_argument('-with_tll_on_img', type=int, default=0)

   
   

    opt = parser.parse_args()
    opt.data = f"data/{opt.data}/"
    # default device is CUDA
    opt.device = torch.device('cuda')
    
    opt.data_name = opt.data.split('/')[-2]

    if opt.method == "mle":
        opt.model_saved_name = f'/{opt.model}_{opt.method}_numgrid{opt.num_grid}_{opt.data_name}_{opt.with_survival}_epoch{opt.epoch}_{opt.seed}'
    elif opt.method == "wsm":
        opt.model_saved_name = f'/{opt.model}_{opt.method}_{opt.data_name}_{opt.with_survival}_alpha{opt.CE_coef}_epoch{opt.epoch}_{opt.seed}'
    elif opt.method == "dsm":
        opt.model_saved_name = f'/{opt.model}_{opt.method}_{opt.data_name}_{opt.with_survival}_alpha{opt.CE_coef}_noise{opt.noise_var}_num{opt.num_noise}_epoch{opt.epoch}_{opt.seed}'
    opt.results_saved_path = 'results{dir_name}'.format(dir_name=opt.model_saved_name) # IMPORTANT: save model, logging, plots here
    if not os.path.exists(opt.results_saved_path):
        os.makedirs(opt.results_saved_path)
    logging.basicConfig(filename='{results_path}/{logging_name}.log'
                                .format(results_path=opt.results_saved_path,logging_name=opt.model_saved_name), 
                                    format="%(asctime)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s", 
                                    level=logging.DEBUG, filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler()) # show on console
    logging.getLogger('matplotlib.font_manager').disabled = True # disable matplotlib logging

    logging.info('parameters: {}'.format(opt))

    """ prepare dataloader """
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    trainloader, testloader, num_types, max_time = prepare_dataloader(opt)
    opt.num_types =num_types
    """ prepare model """

    if opt.model == 'thp_transformer':
        model = thp_Transformer(
            num_types=num_types,
            d_model=opt.d_model,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
            opt = opt,
            max_time = max_time
        )
        for param in model.parameters():
            param.data = param.data.double()
    elif opt.model == "sahp":
        model = SAHP(model_config=opt, max_time=max_time)
        for param in model.parameters():
            param.data = param.data.double()


    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """

    pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Number of parameters: {}'.format(num_params))


    if opt.load_model:
        model.load_state_dict(torch.load(opt.results_saved_path + opt.model_saved_name + '.pth'))
        # model.load_state_dict(torch.load('results/sahp_wsm_taxi_1_alpha10.0_epoch250_1/sahp_wsm_taxi_1_alpha10.0_epoch250_1.pth'))
        logging.info("loading models from {}".format(opt.results_saved_path + opt.model_saved_name + '.pth'))
    """ train the model """

    if opt.train_able:
            train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)
    

    opt.train_able=False
    model.load_state_dict(torch.load(opt.results_saved_path + opt.model_saved_name + '.pth'))
    tll = Utils.eval_loglikelihood(model, testloader, opt)
    acc = Utils.eval_accuracy(model, testloader, opt)

    if not opt.with_tll_on_img:
        Utils.eval_intensity(model, testloader, opt)
    else:
        Utils.eval_intensity(model, testloader, opt, tll)
    logging.info("The testing tll and accuracy are {} and {}".format(tll, acc))

    



if __name__ == '__main__':
    main()
