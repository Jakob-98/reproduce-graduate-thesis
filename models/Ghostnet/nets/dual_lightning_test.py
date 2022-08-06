# %%
import dataloaders
# from ghostnet import ghostnet
import dual_ghostnet
import dual_dataloaders
from importlib import reload
import experiment
# reload(dataloaders)
# reload(ghostnet)
# reload(experiment)
# from experiment import Experiment
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

########################################################################################################################
# standard argparser for config
########################################################################################################################

import argparse, yaml

parser = argparse.ArgumentParser(description='Configure a jconfig')
parser.add_argument('-c', '--config', help='Config .yaml file path', type=str, default='./config.yaml')
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configyaml = yaml.load(f, Loader=yaml.FullLoader)

class config:
    for key, value in configyaml.items():
        locals()[key] = value

print(config.image_path1)
########################################################################################################################
# main

wandb_logger = WandbLogger()
datamodule = dual_dataloaders.DataModuleCustom(
    trainhistlbppath=config.histlbp_path, trainimagepath2=config.image_path2,
    trainimagepath1=config.image_path1, trainlabelpath=config.label_path,
    valhistlbppath=config.val_histlbp_path, valimagepath1=config.val_image_path1,
    valimagepath2=config.val_image_path2, vallabelpath=config.val_label_path,
    nclass=config.nclass)
# %%
model = dual_ghostnet.DualGhostNet(num_classes=config.nclass)
model.load_state_dict(torch.load(config.ptpath))
model.eval()
loss = nn.CrossEntropyLoss()
ex = experiment.Experiment(model, loss, config.nclass, dual_images=True)
trainer = pl.Trainer(max_epochs=1, accelerator='gpu', logger=wandb_logger)
# %%
datamodule.setup()
trainer.validate(ex, dataloaders=datamodule)
# torch.save(model.state_dict(), './ghost_model.pt')
# %%

