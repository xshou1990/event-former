import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

import transformer.Constants as Constants
import Utils

from preprocess.Dataset_TEST import get_dataloader
from transformer.Pretrain import Transformer
from tqdm import tqdm


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def prepare_dataloader(opt,seed):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

   # print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train_defi_polygon.pkl', 'train')
   # print('[Info] Loading dev data...')
    dev_data, num_types = load_data(opt.data + 'dev_defi_polygon.pkl', 'dev')


    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True, seed=seed)
    devloader = get_dataloader(dev_data, opt.batch_size, shuffle=False, seed=seed)

    return trainloader, devloader, num_types

def batch_ave(real_pred, output):
    # sum of all h_i's --> [batchsize, hidden dimension]
    real_pred_batch_sum = torch.sum(torch.unsqueeze(real_pred, -1).repeat(1,1,len(output[0,0,:])) * output , dim = 1 )
    # for each sequence obtain a sum of all real,prediction h_i's
    real_pred_dimen = torch.unsqueeze(torch.sum(real_pred, dim=1),dim =-1)
    # averaged [batchsize, hidden]
    real_pred_batch_ave = real_pred_batch_sum/(real_pred_dimen+1)
    return real_pred_batch_ave


def cosine_similarity(real_pred_ave,real_nonpred_ave  ):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(real_pred_ave,real_nonpred_ave)
    return similarity

def train_epoch(model, classification, regression,  training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    
    time_loss_tr = 0
    type_loss_tr = 0
    loss_tr = 0
    
    
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        
        """ prepare data """
        event_time, _, event_type, time_label, type_label = map(lambda x: x.to(opt.device), batch)         
        # 1's for masked, 0 for nonmasked, MLM, or difference between actual and masked
        prediction_mask = ((time_label-event_time)!=0)
        ## keep track of event instances that are not padded
        nonpadded_events = time_label > 0 
        ## keep track of event instances that are not void i.e. real
        real_mask =  ( torch.arange(len(prediction_mask[0])) % 2 == 0).repeat(len(prediction_mask),1) .to('cuda')
        
        ## the key four quantities for contrastive loss
        real_pred = ( prediction_mask * real_mask  ) 
        fake_pred = ( prediction_mask * (real_mask == 0) ) 
        real_nonpred = ((prediction_mask ==0) *  real_mask * nonpadded_events ) 
        fake_nonpred = ((prediction_mask ==0) * (real_mask == 0) * nonpadded_events ) 

       
        """ forward """
        optimizer.zero_grad()
        ### output will be H1_,...,H_n 
        output = model(event_type, event_time, type_label)
        ### no need to use non_pad mask since for padd entry 0, time_label-event_time will always be 0
        ### prediction_mask takes care of it

        
        """ label predict loss """
        # given H_i, predict (t_i, y_i) we padded with t_i =0, y_i - numtypes+1
        type_pred = F.softmax(classification(output),dim=-1)
        # only predict number of types, total of num_types 
        type_loss_value = F.cross_entropy(type_pred[prediction_mask,:], type_label[prediction_mask], reduction='mean') 


        """ time predict loss """
        time_pred = torch.exp(torch.squeeze(regression(output)))
        time_loss_value = F.mse_loss(time_pred[prediction_mask], time_label[prediction_mask],reduction='mean')  
        
        
        """ prediction loss """
        loss_pred =  type_loss_value  + opt.gamma* time_loss_value 
        
        """ contrastive loss """
        # the for averaged representations     
        real_pred_ave =  batch_ave(real_pred, output)
        real_nonpred_ave =  batch_ave(real_nonpred, output)
        fake_pred_ave =  batch_ave(fake_pred, output)
        fake_nonpred_ave =  batch_ave(fake_nonpred, output)
        
        
        real_sim = cosine_similarity(real_pred_ave,real_nonpred_ave  ) # real in prediction vs real non in prediction
        fake_sim = cosine_similarity(fake_pred_ave,fake_nonpred_ave  ) # fake in prediction vs fake non in prediction
        real_fake_pred = cosine_similarity(real_pred_ave,fake_pred_ave ) # real in prediction vs fake in prediction
        real_fake_nonpred = cosine_similarity(real_nonpred_ave,fake_nonpred_ave ) # real non in prediction vs fake not in prediction
      
 #         loss_contrast = -torch.mean(torch.log (torch.exp((real_sim + fake_sim)/opt.tau) / torch.exp((real_fake_pred + real_fake_nonpred)/opt.tau)))
        loss_contrast =  torch.mean( (real_fake_pred + real_fake_nonpred) - (real_sim + fake_sim) ) 
     
        
        del real_sim,fake_sim,real_fake_pred,real_fake_nonpred
        del real_pred_ave , real_nonpred_ave, fake_pred_ave, fake_nonpred_ave
        del real_pred , real_nonpred, fake_pred, fake_nonpred
        del time_pred, type_pred,  output
        del prediction_mask, nonpadded_events, real_mask 
        del event_time, event_type, time_label, type_label
        
                
        """ total loss """
        loss = loss_pred + opt.beta * loss_contrast
        

        """ backward """        
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        loss_tr += loss.item()
        time_loss_tr += time_loss_value.item()
        type_loss_tr += type_loss_value.item()

    return loss_tr, time_loss_tr , type_loss_tr

def eval_epoch(model, classification, regression, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    time_loss_val = 0
    type_loss_val = 0
    loss_val = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type, time_label, type_label = map(lambda x: x.to(opt.device), batch)        

            ## keep track of event instances to be predicted
            prediction_mask = ((time_label-event_time)!=0)
            ## keep track of event instances that are not padded
            nonpadded_events = time_label > 0 
            ## keep track of event instances that are not void i.e. real
            real_mask =  ( torch.arange(len(prediction_mask[0])) % 2 == 0).repeat(len(prediction_mask),1).to('cuda')

            ## the key four quantities for contrastive loss
            real_pred = ( prediction_mask * real_mask  ) 
            fake_pred = ( prediction_mask * (real_mask == 0) ) 
            real_nonpred = ((prediction_mask ==0) *  real_mask * nonpadded_events ) 
            fake_nonpred = ((prediction_mask ==0) * (real_mask == 0) * nonpadded_events )   

            """ forward """
            ### output will be H1_,...,H_n
            output = model(event_type, event_time,type_label)

 
 
            """ label predict loss """
            # given H_i, predict y_i 
            type_pred = F.softmax(classification(output),dim=-1)
            # only predict number of types, include void
            type_loss_value = F.cross_entropy(type_pred[prediction_mask,:], type_label[prediction_mask], reduction='mean') 


            """ time predict loss """
            time_pred = torch.exp(torch.squeeze(regression(output)))
            time_loss_value = F.mse_loss(time_pred[prediction_mask], time_label[prediction_mask],reduction='mean')  
            
            loss_pred =  type_loss_value  + opt.gamma*time_loss_value 
            
            """  contrastive loss """
            # the for averaged representations     
            real_pred_ave =  batch_ave(real_pred, output)
            real_nonpred_ave =  batch_ave(real_nonpred, output)
            fake_pred_ave =  batch_ave(fake_pred, output)
            fake_nonpred_ave =  batch_ave(fake_nonpred, output)

            real_sim = cosine_similarity(real_pred_ave,real_nonpred_ave  ) # real in prediction vs real non in prediction
            fake_sim = cosine_similarity(fake_pred_ave,fake_nonpred_ave  ) # fake in prediction vs fake non in prediction
            real_fake_pred = cosine_similarity(real_pred_ave,fake_pred_ave ) # real in prediction vs fake in prediction
            real_fake_nonpred = cosine_similarity(real_nonpred_ave,fake_nonpred_ave ) # real non in prediction vs fake not in prediction

           # loss_contrast = -torch.mean(torch.log (torch.exp((real_sim + fake_sim)/opt.tau) / torch.exp((real_fake_pred + real_fake_nonpred)/opt.tau)))
            loss_contrast =  torch.mean( (real_fake_pred + real_fake_nonpred) - (real_sim + fake_sim) ) 

            
            """ total loss """
            loss = loss_pred + opt.beta * loss_contrast
            
            del real_sim,fake_sim,real_fake_pred,real_fake_nonpred
            del real_pred_ave , real_nonpred_ave, fake_pred_ave, fake_nonpred_ave
            del real_pred , real_nonpred, fake_pred, fake_nonpred
            del time_pred, type_pred,  output
            del prediction_mask, nonpadded_events, real_mask 
            del event_time, event_type, time_label, type_label
            
            """ note keeping """
            loss_val += loss.item()
            time_loss_val += time_loss_value.item()
            type_loss_val += type_loss_value.item()
     
    return loss_val, time_loss_val, type_loss_val


def train(model, classification, regression,  training_data, validation_data, optimizer, scheduler, opt):
    """ Start training. """

    valid_event_losses = [] 
    valid_time_losses = []
    valid_type_losses = []
    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')
    
        start = time.time()
        #  random insert void epochs with frequency of every 20 epochs
#         if epoch % 20 == 0:
#             training_data, validation_data, num_types = prepare_dataloader(opt,epoch_i)
        train_event, train_time, train_type = train_epoch(model, classification,regression,  training_data, optimizer,  opt)
        print('  - (Training)    loss_tot: {ll: 8.5f}, time_loss:{lt: 8.5f}, type_loss:{lp: 8.5f}, elapse:{elapse: 1f}'
              .format(ll=train_event, lt=train_time, lp= train_type, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_time, valid_type = eval_epoch(model, classification, regression,  validation_data, opt)
        print('  - (Validation)    loss_tot : {ll: 8.5f}, time_loss:{lt: 8.5f}, type_loss:{lp: 8.5f},  elapse:{elapse: 1f} '
              .format(ll=valid_event, lt=valid_time, lp= valid_type, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_time_losses += [valid_time]
        valid_type_losses += [valid_type]


        print('  - [Info] Minimum valid loss w/ gamma of {gamma: 1f},  {event: 8.5f}, '
              ' minimum MSE: {mse: 8.5f} , minium CE : {pred: 8.5f} '
              .format(gamma=opt.gamma, event=min(valid_event_losses),  mse=min(valid_time_losses), pred=min(valid_type_losses)))

        # logging
        with open(opt.log, 'a') as f:
            f.write(' {gamma: 1f}, {epoch}, {ll: 8.5f},  {lp: 8.5f} , {lt: 8.5f} \n'
                    .format( gamma=opt.gamma, epoch=epoch, ll=valid_event, lp=valid_time, lt=valid_type))

        if (best_loss - valid_event) < 1e-4:
            impatient += 1
            if valid_event < best_loss:
                best_loss = valid_event
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = valid_event
            best_model = deepcopy(model.state_dict())
            impatient = 0

        if impatient >= 5:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break
            
        scheduler.step()
        
    return best_model




def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-save_model', required=True)
#     parser.add_argument('-pretrain', type=int, default=1)

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=4)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-gamma', type=float, default=1)
    parser.add_argument('-beta', type=float, default=10)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')
    
    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Gamma, Epoch, valid_event, valid_time, valid_type  \n')
    
    print('[Info] parameters: {}'.format(opt))
    
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    """ prepare dataloader """
    training_data, validation_data, num_types = prepare_dataloader(opt,0)
    
    regression = nn.Linear(opt.d_model, 1, bias=True)
    classification = nn.Linear(opt.d_model, num_types+2, bias=True)
    regression.to(opt.device)
    classification.to(opt.device)


    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_inner=opt.d_inner,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)


    params = list(model.parameters()) + list(regression.parameters()) + list (classification.parameters()) 

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, params),
                            opt.lr, betas=(0.9, 0.999), eps=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)


    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))



    """ train the model """
    best_model = train(model,classification,regression, training_data, validation_data, optimizer, scheduler, opt)
    model.load_state_dict(best_model)
#     torch.save(model.state_dict(), opt.save_model + "batchsize{}gamma{}woh".format(opt.batch_size, opt.gamma))
    torch.save(model, opt.save_model)




if __name__ == '__main__':
    main()

