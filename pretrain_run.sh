device=0
data=datastore/gems/pgem/
save_model=saved_models/pgem/2000woe/pretrain_novoid_batch32_mask15_1001 ## path + name of saved model
batch=32
n_head=4
n_layers=4
d_model=512
d_inner=1024
d_k=512
d_v=512
gamma=1
beta=0.01
dropout=0.1
lr=1e-4
epoch=100
log=pretrain_test.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Pretrain_Main.py -data $data -save_model $save_model -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -gamma $gamma -beta $beta -dropout $dropout -lr $lr  -epoch $epoch -log $log


