# SELF-SUPERVISED CONTRASTIVE PRE-TRAINING FOR MULTIVARIATE EVENT STREAMS

Source code for SELF-SUPERVISED CONTRASTIVE PRE-TRAINING FOR MULTIVARIATE EVENT STREAMS

# Run the code for  SELF-SUPERVISED CONTRASTIVE PRE-TRAINING FOR MULTIVARIATE EVENT STREAMS

### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.8.0. above

### Instructions
1. folders:
a. targetdata. This folder contains target datasets. Each dataset contains train dev test pickled files. 
b.preprocess. We used to convert data into standard inpt for our model.
c.transformer. This folder contains our main models, modules that supports the training of our models.
d. datastore. This folder contains datasets for pretraining. Each dataset contains train dev pickled files. 
e. saved_models. This folder is a place holder for save models in pretraining and fine tuning. 

2. **bash run.sh** to run the code (i.e.  ./so_run.sh ). 

3. Caution: to run each dataset, change the transformer/Constants.py to appropriate number of types.

4.Additional datasets are available at 
[Google Drive Repository 1](https://drive.google.com/drive/folders/1gt4cR-yFINO745bcC5CMenwZ9QWGfFgr) 
and 
[Google Drive Repository 2](https://drive.google.com/drive/folders/1K46x1NiaSuKEhWkEkFe5avYr3ltai_6R).


