device=0
data="data/stackoverflow/"
train_able=1
batch=64
n_head=1
n_layers=1
d_model=16
d_inner=8
d_k=16
d_v=16
dropout=0.1
lr=1e-3
smooth=0.1
epoch=100

method='mle'
num_grid=10
seq_trunc=0
delete_outlier=0
log=log.txt
model='sahp'
load_model=0
seed=1
cd ../..
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -train_able $train_able -model $model -num_grid $num_grid -seq_trunc $seq_trunc -delete_outlier $delete_outlier -method $method    -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log -seed $seed -load_model $load_model
