# device=0
# data=targetdata/hawkes/hawkes_exp/sample_5/
# pretrain_model=saved_models/hawkes_exp/2000woh/pretrain_novoid_batch32_mask15_11 ## pretrain model name
# save_model=saved_models/hawkes_exp/pretrained/2000woh/s5_transfer_novoid_batch32_mask15_11
# batch=32
# lr=2e-3
# epoch=100
# log=transfer_hawkes.txt

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Transfer_Main.py -data $data -pretrain_model $pretrain_model -save_model $save_model -batch $batch -lr $lr  -epoch $epoch -log $log


device=0
data=targetdata/gems/pgem/sample_5/
pretrain_model=saved_models/pgem/2000woe/pretrain_rand_batch32_mask15_11 ## pretrain model name
save_model=saved_models/pgem/pretrained/2000woe/s5_transfer_rand_batch32_mask15_11_2lr
batch=32
lr=2e-3
epoch=100
log=transfer_gems.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Transfer_Main.py -data $data -pretrain_model $pretrain_model -save_model $save_model -batch $batch -lr $lr  -epoch $epoch -log $log