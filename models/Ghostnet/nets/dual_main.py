# %%
import traceback
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch.nn as nn
import dual_dataloaders
# from ghostnet import ghostnet
import dual_ghostnet
from importlib import reload
import experiment
# reload(dual_dataloaders)
# reload(dual_ghostnet)
# reload(experiment)
# from experiment import Experiment


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

########################################################################################################################
# main

# class config:
#     image_path1 = "C:/temp/ispipeline/images/224xCropRGBTrain5/"
#     label_path = "C:/temp/ispipeline/labels/224xCropRGBTrain5/"
#     histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBTrain5/"
#     val_image_path1 = "C:/temp/ispipeline/images/224xCropRGBval20"
#     val_label_path = "C:/temp/ispipeline/labels/224xCropRGBval20/"
#     val_histlbp_path = "C:/temp/ispipeline/histlbp/224xCropRGBval20/"
#     image_path2 = "C:/temp/ispipeline/images/224xSeqRGBTrain5/"
#     # label_path = "C:/temp/ispipeline/labels/224xSeqRGBTrain5/"
#     # histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBTrain5/"
#     val_image_path2 = "C:/temp/ispipeline/images/224xSeqRGBval20"
#     # val_label_path = "C:/temp/ispipeline/labels/224xSeqRGBval20/"
#     # val_histlbp_path = "C:/temp/ispipeline/histlbp/224xSeqRGBval20/"
#     image_size = 224
#     nclass = 6


wandb_logger = WandbLogger()
datamodule = dual_dataloaders.DataModuleCustom(
    trainhistlbppath=config.histlbp_path, trainimagepath2=config.image_path2,
    trainimagepath1=config.image_path1, trainlabelpath=config.label_path,
    valhistlbppath=config.val_histlbp_path, valimagepath1=config.val_image_path1,
    valimagepath2=config.val_image_path2, vallabelpath=config.val_label_path,
    nclass=config.nclass)

# %%
model = dual_ghostnet.DualGhostNet(num_classes=config.nclass)

model.eval()
loss = nn.CrossEntropyLoss()
ex = experiment.Experiment(model, loss, config.nclass, dual_images=True)
trainer = pl.Trainer(max_epochs=config.max_epochs, accelerator='gpu', logger=wandb_logger)
# %%
datamodule.setup()
trainer.fit(ex, train_dataloaders=datamodule)
torch.save(model.state_dict(), f'./{config.ex_name}_final_model.pt')
# %%
