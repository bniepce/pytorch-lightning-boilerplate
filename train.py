import pytorch_lightning as pl
from argparse import ArgumentParser
from src.utils.config import ConfigParser
import src.dataloaders as dataloader_module
import src.models as model_module
from pytorch_lightning import loggers as pl_loggers

if __name__ == '__main__':

    pl.seed_everything(1234)

    args = ArgumentParser()
    args.add_argument('--config', default = './config/mnist.json', type = str)
    config = ConfigParser.from_args(args)

    datamodule = config.init_obj('dataloader', dataloader_module)
    model = config.init_obj('model', model_module)
    epochs = config.__getitem__('trainer')['epochs']
    accelerator = config.__getitem__('accelerator')

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(accelerator=accelerator, max_epochs=epochs, logger=tb_logger)
    trainer.fit(model, datamodule)
    trainer.test(datamodule=datamodule, ckpt_path='best')