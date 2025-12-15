
from scipy.stats import expon
from scipy.stats import uniform
from scipy.stats import multinomial
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.linalg import sqrtm

import pickle as pkl

# def intensity(t,m,history,M,parameters):   #left continue    beta_12 is influence from 2 to 1
#     mu=parameters[0][m]   #value
#     alpha=parameters[1][m]   #vector
#     intensity=0
#     for n in range(M):
#         for i in range(len(history[n])):
#             if history[n][i]>=t:
#                 break
#             else:
#                 intensity+=alpha[n]*np.exp(np.sin((t-history[n][i])))
#     return mu+intensity

def intensity(t,m,history,M,parameters):   #left continue    beta_12 is influence from 2 to 1
  mu=parameters[0][m]  #value

  alpha=parameters[1][m]  #vector

  intensity=0

  for n in range(M):
    for i in range(len(history[n])):
        if history[n][i]>=t:
            break
        else:
            intensity+=alpha[n]*np.exp(-5. * (t-history[n][i]))

  return mu+intensity

def multi_hawkes_simulation(mu,alpha,M,T):
    assert len(mu)==len(alpha[0])==M
    t=0
    parameters=[mu,alpha]
    jump= [0]*M
    points_hawkes=[]
    for i in range(M):
        points_hawkes.append([])
    while(t<T):
        intensity_sup=sum(intensity(t,m,points_hawkes,M,parameters)+jump[m] for m in range(M))
        r=expon.rvs(scale=1/intensity_sup) #scale=1/lamda
        t+=r
        D=uniform.rvs(loc=0,scale=1)
        if D*intensity_sup<=sum(intensity(t,m,points_hawkes,M,parameters) for m in range(M)):
            sum_intensity=sum(intensity(t,m,points_hawkes,M,parameters) for m in range(M))
            k=list(multinomial.rvs(1,[intensity(t,m,points_hawkes,M,parameters)/sum_intensity for m in range(M)])).index(1)
            points_hawkes[k].append(t)
            jump=np.array(alpha)[:,k]
        else:
            jump=np.array(alpha)[:,k]
    if points_hawkes[k][-1]>T:
        del points_hawkes[k][-1]
    return points_hawkes

#################################### Simuliation ####################################
n_realizations=2000
T=20
M=2
# mu=[1.,1.5]
# alpha=[[0.33,0.1],[0.05,0.33]]
mu=[2.0,2.5]
alpha=[[0.33,0.1],[0.05,0.33]]
parameters=[mu,alpha]

print(mu)

time_seqs=[]
type_seqs = []
for seq_idx in range(n_realizations):
    print('{}-th seqs'.format(seq_idx))

    points_hawkes=multi_hawkes_simulation(mu,alpha,M,T)

    results = []
    for m in range(M):
        time=np.array(points_hawkes[m]).reshape(-1,1)
        type = np.ones_like(time)*m


        concatenation = np.concatenate((time, type), axis=-1)

        if len(results) == 0:
            results  = concatenation
        else:
            results = np.concatenate((concatenation, results), axis=0)

    results = results[results[:, 0].argsort()]

    time_seqs.append(results[:,0])
    type_seqs.append(results[:,1])

#################################### Split, Format, Save ####################################
tr_len, dev_len, test_len = int(n_realizations *(0.5)), int(n_realizations *(0.25)), int(n_realizations *(0.25))

tr_times = time_seqs[:tr_len]
dev_times = time_seqs[tr_len:tr_len+dev_len]
test_times = time_seqs[tr_len+dev_len:]

tr_types = type_seqs[:tr_len]
dev_types = type_seqs[tr_len:tr_len+dev_len]
test_types = type_seqs[tr_len+dev_len:]

def format_converter(times, type):
    batch_ = len(times)
    results = []

    for seq_idx in range(batch_):
        seq_time = np.array(times[seq_idx])
        seq_type = np.array(type[seq_idx])
        
        # non_padding_mask = seq_time != 0
        # seq_time = seq_time[non_padding_mask]
        # seq_type = seq_type[non_padding_mask]

        seq_diff = [seq_time[0]] +list(seq_time[1:] - seq_time[:-1])
        result_seq = []
        for i in range(len(seq_time)):
            event = {'time_since_start': seq_time[i], 'time_since_last_event':seq_diff[i], 'type_event': seq_type[i]}
            result_seq.append(event)
        results.append(result_seq)
    return results

tr_results = format_converter(tr_times, tr_types)
dev_results = format_converter(dev_times, dev_types)
test_results = format_converter(test_times, test_types)

_to_dump_train_ = {'train': tr_results, 'test':[], 'dev':[], 'dim_process':M}
_to_dump_test_ = {'train': [], 'test':test_results, 'dev':[], 'dim_process': M}
_to_dump_dev_ = {'train': [], 'test':[], 'dev':dev_results, 'dim_process': M}

with open('train.pkl', 'wb') as f:
    pkl.dump(_to_dump_train_, f)

with open('test.pkl', 'wb') as f:
    pkl.dump(_to_dump_test_, f)

with open('dev.pkl', 'wb') as f:
    pkl.dump(_to_dump_dev_, f)

visualize = True
if visualize:
    plt.figure(1,figsize=(7,5))
    plt.subplot(1,1,1)             # points position
    for m in range(M):
        plt.plot([0,T],[m,m],'r-',lw=1,alpha=0.6)
        plt.plot(points_hawkes[m],[m]*len(points_hawkes[m]),linestyle='None', marker='|', markersize=10,label=('hawkes %s'%(m+1)))
    plt.title('multivariate hawkes process')
    plt.ylim(-1,2)
    plt.legend(loc='best',frameon=0)
    plt.savefig('hawkes points.png')

    x=np.linspace(0,T,2000)    # intensity function curve
    y=[[]]*M
    for m in range(M):
        y[m]=np.fromiter([intensity(xi,m,points_hawkes,M,parameters) for xi in x],np.float)
    plt.figure(1,figsize=(10,5)) 
    for m in range(M):
        plt.subplot(M,1,m+1)
        plt.plot(x,y[m],'r-',label=('hawkes %s intensity'%(m+1)))
        plt.ylim(0)
        plt.legend(loc='best',frameon=0)
    plt.savefig('multi-hawkes intensity.png')



