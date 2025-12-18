import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Pretrain import Transformer
from transformer.Predictor import Predictor_Net

import sys
sys.argv=['']
del sys


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading test data...')
    test_data, num_types = load_data(opt.data + 'test_bangladesh.pkl', 'test')

    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return testloader, num_types



def eval_epoch(model, pretrain_model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

#     total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            _, time_gap,_, event_time, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            
            
            enc_out = pretrain_model(event_type, event_time, event_type)
#             enc_out = pretrain_model(event_type, event_time)
            
            prediction = model( enc_out, event_type)
            

            """ compute loss """
#             event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
#             event_loss = -torch.sum(event_ll - non_event_ll)
            
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)
            
#             scale_time_loss = 1
#             loss = event_loss + pred_loss + se / scale_time_loss

            """ note keeping """
#             total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_rate / total_num_pred, rmse



parsed_args = argparse.ArgumentParser()
parsed_args.device = 0
parsed_args.data = "targetdata/ACLED_bangladesh/"
parsed_args.pretrain_model = "saved_models/ACLED_bangladesh/pretrain_rand_batch64_mask15_1001"  # pretrain model
parsed_args.save_model="saved_models/ACLED_bangladesh/pretrained/transfer_rand_batch64_mask15_1001" # pretrain model dict
parsed_args.batch_size = 1
parsed_args.smooth = 0.1



opt = parsed_args

# default device is CUDA
opt.device = torch.device('cuda')

np.random.seed(0)
torch.manual_seed(0)

""" prepare dataloader """
testloader, num_types = prepare_dataloader(opt)

""" load pretrain model """
pretrain_model = torch.load(opt.pretrain_model,map_location='cuda')
pretrain_model.to(opt.device)


""" prepare model """
model = Predictor_Net(
    num_types=num_types,
    d_model=pretrain_model.encoder.d_model
)
model.to(opt.device)
best_model= torch.load(opt.save_model, map_location='cuda')
model.load_state_dict(best_model)


if opt.smooth > 0:
    pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
else:
    pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

# opt.batch_size = 500
te,_= prepare_dataloader(opt)

accuracy,rmse = eval_epoch(model,  pretrain_model, te, pred_loss_func, opt)
print(" total_event_rate / total_num_pred {}, rmse {}".format(accuracy,rmse))

