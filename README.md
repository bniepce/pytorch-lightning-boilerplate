<div align="center">    
 
# Pytorch Lightning Boilerplate  

This repository is a WIP.

</div>
 
## Description   

This repo contains a boilerplate of a Pytorch Lightning project based on the [pytorch-template](https://github.com/victoresque/pytorch-template) and the [deep-learning-project-template](https://github.com/PyTorchLightning/deep-learning-project-template).

## Usage 
First, install dependencies
```bash
# Clone project
git clone https://github.com/bniepce/pytorch-lightning-boilerplate.git

# Setup project
cd pytorch-lightning-boilerplate
pip install -e .   
pip install -r requirements.txt

python train.py    
```

## Use custom models

To train another model than the MNISTModel provided in this repo, one just has to provide the computational graph of that model and define a new configuration file under the config folder.