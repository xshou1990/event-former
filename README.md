# Self-Supervised Contrastive Pre-Training for Multivariate Temporal Point Processes

Source code for Self-Supervised Contrastive Pre-Training for Multivariate Temporal Point Processes

# Run the code for  Self-Supervised Contrastive Pre-Training for Multivariate Temporal Point Processes

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

2. **bash run.sh** to run the code (i.e.  ./so_run.sh ). If permission denied, put chmod u+x so_run.sh on commandline. and then ./so_run.sh.  See discussion https://stackoverflow.com/questions/18960689/ubuntu-says-bash-program-permission-denied

3. Caution: to run each dataset, change the transformer/Constants.py to appropriate number of types.