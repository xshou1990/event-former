import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer.Constants as Constants

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear1 = nn.Linear(dim, dim, bias=True)
        self.linear2 = nn.Linear(dim, dim , bias=True)
        self.linear3 = nn.Linear(dim, num_types, bias=True)
        

        self.relu = torch.nn.ReLU()
        

    def forward(self, data, non_pad_mask):
        out = self.linear1(data)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        
        out = out * non_pad_mask
        
        return out
  


class Predictor_Net(nn.Module):
    def __init__(
        self,
        num_types, d_model=512):
        super().__init__()
    
        self.num_types = num_types

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model,1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)
    
    
    def forward(self, enc_output, event_type):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input:  enc_output: batch*seq_len*model_dim; (pretrained)
               event_time: batch*seq_len.
        Output:  type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        non_pad_mask = get_non_pad_mask(event_type)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return (type_prediction, time_prediction)


    
    