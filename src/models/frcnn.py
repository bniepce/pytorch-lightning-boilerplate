
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(pl.LightningModule):
    def __init__(self, num_classes = 5, learning_rate = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = [{k: v for k, v in t.items()} for t in y]
        loss_dict = self.model(x, y)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses)
        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = [{k: v for k, v in t.items()} for t in y]
        outputs = self.model(x, y)
        res = {target["image_id"].item(): output for target, output in zip(y, outputs)}
        return {}
    
    def validation_epoch_end(self, outputs):
        metric = 0
        tensorboard_logs = {'main_score': metric}
        return {'val_loss': metric, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.95, weight_decay=1e-5, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0, verbose=True)
        return [optimizer], [scheduler]