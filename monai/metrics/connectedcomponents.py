# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Callable, Optional, Union

import torch
import numpy as np
from skimage import measure
import scipy
import os
import pandas as pd
import sys

from monai.networks import one_hot
from monai.utils import MetricReduction


class ConnectedComponentMetric:
    """
    TODO.

    Args:
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        mutually_exclusive: if True, `y_pred` will be converted into a binary matrix using
            a combination of argmax and to_onehot.  Defaults to False.
        sigmoid: whether to add sigmoid function to y_pred before computation. Defaults to False.
        other_act: callable function to replace `sigmoid` as activation layer if needed, Defaults to ``None``.
            for example: `other_act = torch.tanh`.
        logit_thresh: the threshold value used to convert (for example, after sigmoid if `sigmoid=True`)
            `y_pred` into a binary matrix. Defaults to 0.5.
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.

    Raises:
        ValueError: When ``sigmoid=True`` and ``other_act is not None``. Incompatible values.

    """

    def __init__(
        self,
        sigmoid: bool = False,
        other_act: Optional[Callable] = None,
        logit_thresh: float = 0.5,
        return_val: int =  3,
        log_file: str = ""
    ) -> None:
        super().__init__()
        if sigmoid and other_act is not None:
            raise ValueError("Incompatible values: ``sigmoid=True`` and ``other_act is not None``.")
        self.sigmoid = sigmoid
        self.other_act = other_act
        self.logit_thresh = logit_thresh
        self.return_val = return_val
        self.log_file = log_file
        self.not_nans: Optional[torch.Tensor] = None  # keep track for valid elements in the batch

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                it must be one-hot format and first dim is batch.
            y: ground truth to compute mean dice metric, the first dim is batch.

        Raises:
            ValueError: When ``self.reduction`` is not one of
                ["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].

        """
        f = compute_image_metrics(
            y_pred=y_pred,
            y=y,
            sigmoid=self.sigmoid,
            other_act=self.other_act,
            logit_thresh=self.logit_thresh,
            return_val = self.return_val,
            log_file=self.log_file,
        )
        return f

def connectedComponents(label,sizes_need= False):
    sizes= []
    labelled_label,num = measure.label(input=label,return_num=True,connectivity=2)
    regionprops = measure.regionprops(labelled_label)
    for props in regionprops:
        sizes.append(props.area)
    mass = scipy.ndimage.center_of_mass(label,labelled_label,range(1,num+1))
    listofCenters = []
    for (x,y,z) in mass:
        listofCenters.append([np.rint(x),np.rint(y),np.rint(z)])
    if sizes_need:
        return labelled_label, num,sizes

    return labelled_label,num
#
def get_image_metrics(seg_out, label):
    image_tp = 0
    labelled_seg, seg_num, sizes = connectedComponents(seg_out, sizes_need = True)
    labelled_label, num_gt = connectedComponents(label)
    found_cmbs = []
    overlap = label * labelled_seg
    to_look_at_seg= [uniq for uniq in np.unique(overlap) if not uniq == 0]
    image_fp = seg_num - len(to_look_at_seg)
    for blob_num in to_look_at_seg:
        seg_mask = np.where(labelled_seg == blob_num, 1, 0)
        cmbs_to_consider = [uniq for uniq in np.unique(seg_mask * labelled_label) if not uniq == 0]
        if(len(cmbs_to_consider) > 1):
            overlaps = [np.divide(np.sum(np.where(labelled_label == cmb_,1,0)*seg_mask),np.sum(seg_mask)) for cmb_ in cmbs_to_consider]
            cmb_ = cmbs_to_consider[np.argmax(overlaps)]
        else:
            cmb_ = cmbs_to_consider[0]
        cmb_mask = np.where(labelled_label == cmb_,1,0)
        if np.divide(np.sum(seg_mask * cmb_mask),np.sum(cmb_mask)) >= 0.1 and cmb_ not in found_cmbs:
            image_tp+=1
            found_cmbs.append(cmb_)
        else:
            image_fp+=1
    found_cmbs = [uniq for uniq in np.unique(found_cmbs) if not uniq == 0]
    image_fn = num_gt - len(found_cmbs)
    return image_tp, image_fp, image_fn, num_gt


def compute_image_metrics(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    sigmoid: bool = False,
    other_act: Optional[Callable] = None,
    logit_thresh: float = 0.5,
    return_val: int = 3,
    log_file: str= "",
) -> torch.Tensor():
    """Computes Dice score metric from full size Tensor and collects average.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            it must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
        y: ground truth to compute mean dice metric, the first dim is batch.
            example shape: [16, 1, 32, 32] will be converted into [16, 3, 32, 32].
            alternative shape: [16, 3, 32, 32] and set `to_onehot_y=False` to use 3-class labels directly.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        mutually_exclusive: if True, `y_pred` will be converted into a binary matrix using
            a combination of argmax and to_onehot.  Defaults to False.
        sigmoid: whether to add sigmoid function to y_pred before computation. Defaults to False.
        other_act: callable function to replace `sigmoid` as activation layer if needed, Defaults to ``None``.
            for example: `other_act = torch.tanh`.
        logit_thresh: the threshold value used to convert (for example, after sigmoid if `sigmoid=True`)
            `y_pred` into a binary matrix. Defaults to 0.5.

    Raises:
        ValueError: When ``sigmoid=True`` and ``other_act is not None``. Incompatible values.
        TypeError: When ``other_act`` is not an ``Optional[Callable]``.
        ValueError: When ``sigmoid=True`` and ``mutually_exclusive=True``. Incompatible values.

    Returns:
        Dice scores per batch and per class, (shape [batch_size, n_classes]).

    Note:
        This method provides two options to convert `y_pred` into a binary matrix
            (1) when `mutually_exclusive` is True, it uses a combination of ``argmax`` and ``to_onehot``,
            (2) when `mutually_exclusive` is False, it uses a threshold ``logit_thresh``
                (optionally with a ``sigmoid`` function before thresholding).

    """
    n_classes = y_pred.shape[1]
    if sigmoid and other_act is not None:
        raise ValueError("Incompatible values: sigmoid=True and other_act is not None.")
    if sigmoid:
        y_pred = y_pred.float().sigmoid()

    if other_act is not None:
        if not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        y_pred = other_act(y_pred)

    if n_classes == 1:
        y_pred = (y_pred >= logit_thresh).float()
        y = (y > 0).float()
    else:  # multi-channel y_pred
        raise NotImplementedError("Non binary segmentation not yet supported.")
    assert y.shape == y_pred.shape, "Ground truth one-hot has differing shape (%r) from source (%r)" % (
        y.shape,
        y_pred.shape,
    )
    y = y.float()
    y_pred = y_pred.float()
    y_s = torch.unbind(y, dim = 0)
    y_preds = torch.unbind(y_pred, dim = 0)
    TP = []
    FN = []
    FP = []
    amount_of_lesions_in_gt = []
    for (y_, y_pred_) in zip(y_s,y_preds):
        image_tp, image_fp, image_fn, image_lesions = get_image_metrics(torch.squeeze(y_pred_).detach().cpu().numpy(),torch.squeeze(y_).detach().cpu().numpy())
        TP.append(image_tp)
        FP.append(image_fp)
        FN.append(image_fn)
        amount_of_lesions_in_gt.append(image_lesions)

    data_FP = np.mean(FP)
    data_TP = np.sum(TP) / np.sum(amount_of_lesions_in_gt)
    data_FN = np.sum(FN) / np.sum(amount_of_lesions_in_gt)
    data_precision = np.divide(data_TP, (data_FP + data_TP))
    data_recall = np.divide(data_TP, (data_FN + data_TP))

    return torch.Tensor([data_TP, data_FN, data_FP, data_precision, data_recall])
