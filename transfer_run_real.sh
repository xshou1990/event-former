# device=0
# data=targetdata/ACLED_bangladesh/
# pretrain_model=saved_models/ACLED_bangladesh/pretrain_rand_batch64_mask15_1001
# save_model=saved_models/ACLED_bangladesh/pretrained/transfer_rand_batch64_mask15_1001
# batch=32
# lr=1e-3
# epoch=100
# log=transfer_bangladesh.txt

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Transfer_Main_real.py -data $data -pretrain_model $pretrain_model -save_model $save_model -batch $batch -lr $lr  -epoch $epoch -log $log


# device=0
# data=targetdata/electronics/
# pretrain_model=saved_models/electronics/pretrain_rand_batch64_mask15_1001
# save_model=saved_models/electronics/pretrained/transfer_rand_batch64_mask15_1001
# batch=32
# lr=1e-3
# epoch=100
# log=transfer_electronics.txt

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Transfer_Main_real.py -data $data -pretrain_model $pretrain_model -save_model $save_model -batch $batch -lr $lr  -epoch $epoch -log $log

device=0
data=targetdata/defi_eth/
pretrain_model=saved_models/defi_eth/pretrain_rand_batch32_mask15_1001 ## pretrain model name
save_model=saved_models/defi_eth/pretrained/transfer_rand_batch32_mask15_1001
batch=32
lr=1e-3
epoch=100
log=defi_polygon.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Transfer_Main_real.py -data $data -pretrain_model $pretrain_model -save_model $save_model -batch $batch -lr $lr  -epoch $epoch -log $log

# device=0
# data=targetdata/ACLED_bangladesh/
# pretrain_model=saved_models/ACLED_bangladesh/pretrain_LL ## pretrain model name
# save_model=saved_models/ACLED_bangladesh/pretrained/transfer_LL
# batch=32
# lr=6e-3
# epoch=300
# log=bangladesh_transfer_LL.txt

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Transfer_Main_real_abl.py -data $data -pretrain_model $pretrain_model -save_model $save_model -batch $batch -lr $lr  -epoch $epoch -log $log

# device=0
# data=datastore/ACLED_india_abl/
# pretrain_model=saved_models/ACLED_india_abl/pretrain_rand_batch128_gamma1 ## pretrain model name
# save_model=saved_models/ACLED_india_abl/pretrained/transfer_rand_batch128_gamma1
# batch=32
# lr=8e-3
# epoch=300
# log=bangladesh_transfer_LL.txt

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Transfer_Main_real.py -data $data -pretrain_model $pretrain_model -save_model $save_model -batch $batch -lr $lr  -epoch $epoch -log $log

# device=0
# data=datastore/cosmetics_abl/
# pretrain_model=saved_models/cosmetics_abl/pretrain_rand_batch64_gamma1 ## pretrain model name
# save_model=saved_models/cosmetics_abl/pretrained/transfer_rand_batch64_gamma1_4layer
# batch=32
# lr=4e-3
# epoch=100
# log=cosmetics.txt

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Transfer_Main_real.py -data $data -pretrain_model $pretrain_model -save_model $save_model -batch $batch -lr $lr  -epoch $epoch -log $log