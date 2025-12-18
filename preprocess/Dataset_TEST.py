import numpy as np
import torch
import torch.utils.data

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        
        self.time_labels = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.type_labels = [[elem['type_event'] + 1 for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx],self.time_labels[idx], self.type_labels[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type, time_label,type_label = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    time_label = pad_time(time_label)
    type_label = pad_type(type_label)
    
    return time, time_gap, event_type, time_label,type_label

def MaskEvent(data):
    torch.manual_seed(0)
    ds = EventData(data)

    for i in range(len(ds.event_type)):
        rand = torch.rand(len(ds.event_type[i]))
        # create mask list
        mask_arr = (rand < 0.15) * (ds.event_type != 0)

        selection = torch.flatten(mask_arr.nonzero()).tolist()

        for j in selection: 
            ds.event_type[i][j]  = Constants.PAD 
            ds.time[i][j] = Constants.PAD #num_types + 1
        
    return ds

def MaskEvent_geom(data):
    torch.manual_seed(0)
    ds = EventData(data)

    for i in range(len(ds.event_type)):
        L = np.sum(np.array(ds.event_type[i]) !=0)
        mask_arr = 1-geom_noise_mask_single(L, 3, 0.15)
       

        selection = mask_arr.nonzero()

        for j in selection[0]: 
            ds.event_type[i][j]  = Constants.PAD 
            ds.time[i][j] = Constants.PAD #num_types + 1
        
    return ds


# def MaskEvent_comb(data):
#     torch.manual_seed(0)
#     np.random.seed(0)
#     ds = EventData(data)

#     for i in range(len(ds.event_type)):
#         rand = torch.rand(len(ds.event_type[i]))
#         L = np.sum(np.array(ds.event_type[i]) !=0)
#         time_mask = 1-geom_noise_mask_single(L, 3, 0.3)
# #         print(mask_arr.nonzero())
# #         # create mask list
#         mark_mask = (rand < 0.3) * (ds.event_type != 0)

#         time_selection = time_mask.nonzero()
        
#         mark_selection = torch.flatten(mark_mask.nonzero()).tolist()
# #         print(i,time_selection[0]  )
#         for j in time_selection[0]: 
#             ds.time[i][j] = Constants.PAD #num_types + 1
#         for j in mark_selection: 
#             ds.event_type[i][j]  = Constants.PAD 
        
#     return ds


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    from George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14-18, 2021.
    
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

def voidsamples(data,seed):
    np.random.seed(seed)
    void_list_tot = []
    for j in range(len(data)):
        void_list = []
        for i in range(len(data[j])):
            
#             insert void event at (0, t_1)
#             if i == 0:
#                 void_time = np.random.uniform(low=1e-9, high=data[j][i]['time_since_start'] )
#                 dict_void = {'time_since_start': void_time, 'time_since_last_event': 0.0,'type_event': Constants.num_types }
#                 void_list.append(dict_void)
#             insert void event at (t_i, t_i+1)
            if i != len(data[j])-1:
                void_time = np.random.uniform(low=data[j][i]['time_since_start']+1e-9, high=data[j][i+1]['time_since_start'] )
                dict_void = {'time_since_start': void_time, 'time_since_last_event': 0.0,'type_event': Constants.num_types }
                void_list.append(dict_void)
        new_list = data[j] + void_list     
        sorted_list = sorted(new_list, key=lambda d: d['time_since_start']) 
        void_list_tot.append(sorted_list)
    return void_list_tot

def get_dataloader(data, batch_size, shuffle=True, seed=0):
    """ Prepare dataloader. """
    
    voiddata = voidsamples(data, seed)
    
# #     ds = MaskEvent_geom(voiddata)  # <- masking with void data

    ds = MaskEvent(voiddata)  
    
#     ds = MaskEvent(data)    # no void
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
