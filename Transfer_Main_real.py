import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Predictor import Predictor_Net
from transformer.Pretrain import Transformer
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train_defi_eth_mgn.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev_defi_eth_mgn.pkl', 'dev')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(dev_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, pretrain_model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_pred = 0  # number of predictions
    total_loss = 0 # total loss
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        _, time_gap,_, event_time, event_type = map(lambda x: x.to(opt.device), batch)
        

        """ forward """
        optimizer.zero_grad()

        enc_out = pretrain_model(event_type, event_time, event_type)
        
        prediction = model( enc_out, event_type)
        
        
        """ backward """

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_loss += loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
      
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_loss, total_event_rate / total_num_pred, rmse


def eval_epoch(model, pretrain_model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_pred = 0  # number of predictions
    total_loss = 0 # total loss
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            _, time_gap,_, event_time, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            
            enc_out = pretrain_model(event_type, event_time, event_type)
            
            prediction = model( enc_out, event_type)
            

            """ compute loss """
            
            pred_loss, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time+1e-9)
            
            scale_time_loss = 100
            loss = pred_loss + se / scale_time_loss

            """ note keeping """
            total_loss += loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()

            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return loss, total_event_rate / total_num_pred, rmse


def train(model, pretrain_model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event_epoch, train_type, train_time = train_epoch(model, pretrain_model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)  '
              'tr_loss:{loss: 8.5f}, accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(loss=train_event_epoch,  type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event_epoch, valid_type, valid_time = eval_epoch(model, pretrain_model, validation_data, pred_loss_func, opt)
        print('  - (Dev)    '
              'dev_loss:{loss: 8.5f}, accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(loss=valid_event_epoch, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info]  '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format( pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, acc=valid_type, rmse=valid_time))


        if (best_loss - valid_event_epoch) < 1e-4:
            impatient += 1
            if valid_event_epoch < best_loss:
                best_loss = valid_event_epoch
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = valid_event_epoch
            best_model = deepcopy(model.state_dict())
            impatient = 0

        if impatient >= 10:
            print(f'Breaking due to early stopping at epoch {epoch_i}')
            break
            
        scheduler.step()
        
    return best_model

def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-pretrain_model', required = True)
    parser.add_argument('-save_model', required = True)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, valid_time, valid_type  \n')
    
    print('[Info] parameters: {}'.format(opt))


    np.random.seed(0)
    torch.manual_seed(0)
    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)
    
    print(num_types)

    """ load pretrain model """
    pretrain_model = torch.load(opt.pretrain_model)
    pretrain_model.to(opt.device)

    """ prepare model """
    model = Predictor_Net(
        num_types=num_types,
        d_model=pretrain_model.encoder.d_model
    )
    model.to(opt.device)
    
    
    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    best_model = train(model, pretrain_model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)
    
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), opt.save_model)
#     torch.save(model, opt.save_model + "model_batch{}dmodel{}".format(opt.batch_size,pretrain_model.encoder.d_model))
    


if __name__ == '__main__':
    main()


