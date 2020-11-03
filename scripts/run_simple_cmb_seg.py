#!/usr/bin/env python
# coding: utf-8
import logging
import os
import shutil
import sys
import tempfile
import ignite
import nibabel as nib
import numpy as np
import datetime
import torch
from ignite.engine import _prepare_batch
import sys
# print(sys.path)
# sys.path.append('//home/irme/MONAI/')
from monai.config import print_config
from monai.data import DataLoader, PersistentDataset
from monai.handlers import StatsHandler, CheckpointLoader, CheckpointSaver, \
    TensorBoardStatsHandler, ConnectedComponent, WVVStatsHandler
from ignite.handlers import Checkpoint, DiskSaver
from monai.networks.nets import UNet
from torch.nn import BCEWithLogitsLoss
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadNiftid,
    ScaleIntensityd,
    ToTensord,
    CMBOffsetCropd,

)
from monai.utils import first, argparsing, plot_wvv

print_config()


arguments = argparsing.process()
loss_name = "BCE_logits_2"

channels = (32, 64, 128, 256)
strides = (2, 2, 2)
lr = 1e-4
trainSteps = 10
nChannels = 1
patch_size = (96, 96, 96)
print("Patch size set to: {}".format(patch_size))

nClasses = 1
weight_decay = 0
dropout = 0
dt_string = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")

if not os.path.exists(arguments.save_dir):  # reload an existing directory
    os.mkdir(arguments.save_dir)
# appendix = loss_name + dt_string
appendix = loss_name
save_dir = os.path.join(arguments.save_dir,appendix)


reloading = False
if not os.path.exists(save_dir):  # reload an existing directory
    os.mkdir(save_dir)# create a new save directory
else:

    load_dir = os.path.join(save_dir, "model")
    if not os.path.exists(load_dir):
        print("No model found, resuming normally")
    else:
        reloading = True
        models = sorted(os.listdir(load_dir))
        list_model_time = []
        times = []
        for model_ in models:
            loc = os.path.join(load_dir, model_)
            edit_time = os.path.getmtime(loc)
            times.append(edit_time)
            list_model_time.append([edit_time, loc])
        list_model_time.sort(key=lambda x: x[0])
save_seg_dir = os.path.join(save_dir,'seg_out')
save_seg_dir_tr = os.path.join(save_dir,'seg_out_tr')
save_img_dir = os.path.join(save_dir,'imgs')
save_runs_dir = os.path.join(save_dir,'runs')
print("Model dir: {}".format(save_dir))

if not os.path.exists(save_img_dir):
    os.mkdir(save_img_dir)
if not os.path.exists(save_seg_dir):
    os.mkdir(save_seg_dir)
if not os.path.exists(save_seg_dir_tr):
    os.mkdir(save_seg_dir_tr)

os.environ["MONAI_DATA_DIRECTORY"] = "/home/irme/monai_models"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# # Training data
# image_path0 = '/mnt/data/igroothuis/Data/ALL_DATA/images/fold0'
# image_path1 = '/mnt/data/igroothuis/Data/ALL_DATA/images/fold1'
# image_path2 = '/mnt/data/igroothuis/Data/ALL_DATA/images/fold2'
# label_path0 = '/mnt/data/igroothuis/Data/ALL_DATA/labels/fold0'
# label_path1 = '/mnt/data/igroothuis/Data/ALL_DATA/labels/fold1'
# label_path2 = '/mnt/data/igroothuis/Data/ALL_DATA/labels/fold2'
# # Validation data
# eval_image_path = '/mnt/data/igroothuis/Data/ALL_DATA/images/fold3'
# eval_label_path = '/mnt/data/igroothuis/Data/ALL_DATA/labels/fold3'



image_dirs = ["/home/irme/Data/CleanedData/images/fold0"]
label_dirs = ["/home/irme/Data/CleanedData/labels/fold0"]

image_dir_vals = ["/home/irme/Data/CleanedData/images/fold1"]
label_dir_vals = ["/home/irme/Data/CleanedData/labels/fold1"]

images = []
segs = []

images_val = []
segs_val = []

assert len(image_dirs) == len(label_dirs), "Check training data lists"
assert len(image_dir_vals) == len(label_dir_vals), "Check validation data lists"

for image_dir, label_dir in zip(image_dirs,label_dirs):
    images += [os.path.join(image_dir, image_name) for image_name in
              sorted(os.listdir(image_dir))]
    segs += [os.path.join(label_dir, label_name) for label_name in
            sorted(os.listdir(label_dir))]

data_dicts = [
    {"img": image_name, "seg": label_name}
    for image_name, label_name in zip(images, segs)
]
for image_dir_val, label_dir_val in zip(image_dir_vals,label_dir_vals):
    images_val += [os.path.join(image_dir_val, image_name) for image_name in
                  sorted(os.listdir(image_dir_val))]
    segs_val += [os.path.join(label_dir_val, label_name) for label_name in
                sorted(os.listdir(label_dir_val))]

data_dicts_val = [
    {"img": image_name, "seg": label_name}
    for image_name, label_name in zip(images_val, segs_val)
]



train_transforms = Compose(
    [
        LoadNiftid(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img"]),
        CMBOffsetCropd(keys=['img', 'seg'], image_key='img', label_key='seg',
                       spatial_size=[96, 96, 96]),
        ToTensord(keys=["img", "seg"]),
    ]
)


val_transforms = Compose(
    [
        LoadNiftid(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img"]),
        CMBOffsetCropd(keys=['img', 'seg'], image_key='img', label_key='seg',
                       spatial_size=[96, 96, 96]),
        ToTensord(keys=["img", "seg"]),
    ]
)

ds = PersistentDataset(data=data_dicts, transform=train_transforms, cache_dir="my_cache")
# ds = Dataset(data=data_dicts, transform=train_transforms)
training_loader = DataLoader(ds, batch_size=2, num_workers=4,
                             pin_memory=torch.cuda.is_available())
print(first(training_loader).keys())
# monai.data.PersistentDataset(items, transform=SlowSquare(keys='data'), cache_dir="my_cache")
ds_val = PersistentDataset(data=data_dicts, transform=train_transforms, cache_dir="my_cache")
# ds_val = Dataset(data=data_dicts_val, transform=train_transforms)
val_loader = DataLoader(ds, batch_size=2, num_workers=4,
                        pin_memory=torch.cuda.is_available())
device = torch.device("cuda:0")
net = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

loss = BCEWithLogitsLoss()
lr = 1e-4
opt = torch.optim.Adam(net.parameters(), lr)




def prepare_batch(batch, device=None, non_blocking=False):
    return _prepare_batch((batch['img'].to(device), batch['seg'].to(device)),
                          device, non_blocking)


trainer = ignite.engine.create_supervised_trainer(net, opt, loss, device,
                                                  False,
                                                  prepare_batch=prepare_batch)

to_save = {'model': net, 'optimizer': opt, 'trainer': trainer}


# checkpoint_handler = Checkpoint(to_save,
#                                 DiskSaver(os.path.join(save_dir,'model'),
#                                           create_dir=True,
#                                           require_empty=False),
#                                 '_model',
#                                 n_saved=10)
# trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, checkpoint_handler)
# if reloading:
#     to_load = to_save
#     checkpoint_fp = list_model_time[-1][1]
#     checkpoint = torch.load(checkpoint_fp)
#     checkpoint['trainer']['max_epochs'] = trainSteps
#     Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
#     print("Loaded: {}".format(checkpoint_fp))

log_dir = os.path.join(root_dir, "logs")
checkpoint_saver = CheckpointSaver(save_dir=save_runs_dir, save_dict=to_save, save_interval=1)
checkpoint_saver.attach(trainer)

checkpoint_loader = CheckpointLoader(load_path=save_runs_dir, load_dict=to_save,find_latest= True)
checkpoint_loader.attach(trainer)

# StatsHandler prints loss at every iteration and print metrics at every epoch,
# we don't set metrics for trainer here, so just print loss, user can also customize print functions
# and can use output_transform to convert engine.state.output if it's not a loss value
train_stats_handler = StatsHandler(name="trainer")
train_stats_handler.attach(trainer)

# TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir)
train_tensorboard_stats_handler.attach(trainer)

# optional section for model validation during training
validation_every_n_epochs = 1
# Set parameters for validation

metric_name = "WVV"
log_file = save_dir + "/wwv.csv"
log_file_tr = save_dir + "/wwv_tr.csv"

# add evaluation metric to the evaluator engine
val_metrics = {metric_name: ConnectedComponent(sigmoid=True, return_val=3,
                                               log_file=log_file)}

# Ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
# user can add output_transform to return other values
evaluator = ignite.engine.create_supervised_evaluator(net, val_metrics, device,
                                                      True,
                                                      prepare_batch=prepare_batch)

evaluator_tr = ignite.engine.create_supervised_evaluator(net, val_metrics, device,
                                                      True,
                                                      prepare_batch=prepare_batch)


# create a validation data loader


@trainer.on(
    ignite.engine.Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
def run_validation(engine):
    evaluator.run(val_loader)
    evaluator_tr.run(training_loader)
    plot_wvv.plot_graphs(save_img_dir,wvv=log_file,wvv_tr=log_file_tr)


# Add stats event handler to print validation stats via evaluator
val_stats_handler = WVVStatsHandler(
    name="evaluator",
    output_transform=lambda x: None,
    # no need to print loss value, so disable per iteration output
    global_epoch_transform=lambda x: trainer.state.epoch,
    log_file=log_file
    # fetch global epoch number from trainer
)
val_stats_handler.attach(evaluator)

val_stats_handler = WVVStatsHandler(
    name="evaluator_tr",
    output_transform=lambda x: None,
    # no need to print loss value, so disable per iteration output
    global_epoch_transform=lambda x: trainer.state.epoch,
    log_file=log_file_tr
    # fetch global epoch number from trainer
)
val_stats_handler.attach(evaluator_tr)

# # add handler to record metrics to TensorBoard at every validation epoch
# val_tensorboard_stats_handler = TensorBoardStatsHandler(
#     log_dir=log_dir,
#     output_transform=lambda x: None,
#     # no need to plot loss value, so disable per iteration output
#     global_epoch_transform=lambda x: trainer.state.epoch,
#     # fetch global epoch number from trainer
# )
# val_tensorboard_stats_handler.attach(evaluator)
#
# # add handler to draw the first image and the corresponding label and model output in the last batch
# # here we draw the 3D output as GIF format along Depth axis, at every validation epoch
# val_tensorboard_image_handler = TensorBoardImageHandler(
#     log_dir=log_dir,
#     batch_transform=lambda batch: (batch[0], batch[1]),
#     output_transform=lambda output: predict_segmentation(output[0]),
#     global_iter_transform=lambda x: trainer.state.epoch,
# )
# evaluator.add_event_handler(
#     event_name=ignite.engine.Events.EPOCH_COMPLETED,
#     handler=val_tensorboard_image_handler,
# )

# ## Run training loop

# In[27]:


train_epochs = trainSteps
state = trainer.run(training_loader, train_epochs)

# ## Visualizing Tensorboard logs

# In[ ]:




# Expected training curve on TensorBoard:
# ![image.png](attachment:image.png)

# ## Cleanup data directory
#
# Remove directory if a temporary was used.

# In[ ]:


if directory is None:
    shutil.rmtree(root_dir)
