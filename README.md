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

# Run training
python train.py --config ./config/model.json

# Run inference
python inference.py --config ./config/model.json --model_checkpoint ./logs/logs_ligthning/checkpoint/version__0/model.ckpt
```

A **--data_dir** flag can be used for inference if your test data is not in 

```python
os.path.join(config['dataloader']['args']['data_dir'], 'test')
```

## Use custom models

To train another model than the MNISTModel provided in this repo, one just has to provide the computational graph of that model and define a new configuration file under the config folder.