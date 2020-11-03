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

from typing import Callable, Optional, Sequence

import torch

from monai.metrics import DiceMetric, ConnectedComponentMetric
from monai.utils import MetricReduction, exact_version, optional_import

NotComputableError, _ = optional_import("ignite.exceptions", "0.4.2", exact_version, "NotComputableError")
Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "reinit__is_reduced")
sync_all_reduce, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "sync_all_reduce")


class ConnectedComponent(Metric):  # type: ignore[valid-type, misc] # due to optional_import
    """
    Computes Dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        sigmoid: bool = False,
        other_act: Optional[Callable] = None,
        logit_thresh: float = 0.5,
        output_transform: Callable = lambda x: x,
        device: Optional[torch.device] = None,
        return_val: int = 3,
        log_file: str = "",
    ) -> None:
        """

        Args:
            include_background: whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y: whether to convert the output prediction into the one-hot format. Defaults to False.
            mutually_exclusive: if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False.
            sigmoid: whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            other_act: callable function to replace `sigmoid` as activation layer if needed, Defaults to ``None``.
                for example: `other_act = torch.tanh`.
            logit_thresh: the threshold value to round value to 0.0 and 1.0. Defaults to None (no thresholding).
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.

        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        super().__init__(output_transform, device=device)
        self.cc = ConnectedComponentMetric(
            sigmoid=sigmoid,
            other_act=other_act,
            logit_thresh=logit_thresh,
            return_val= return_val,
            log_file=log_file
        )
        self._tp = 0
        self._fn = 0
        self._fp = 0
        self._precision = 0
        self._recall = 0
        self._num_examples = 0
        self.wvv_results = []

    @reinit__is_reduced
    def reset(self) -> None:
        self._tp = 0
        self._fn = 0
        self._fp = 0
        self._precision = 0
        self._recall = 0
        self._num_examples = 0
        self.wvv_results = []

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        """
        Args:
            output: sequence with contents [y_pred, y].

        Raises:
            ValueError: When ``output`` length is not 2. MeanDice metric can only support y_pred and y.

        """
        if len(output) != 2:
            raise ValueError(f"output must have length 2, got {len(output)}.")
        y_pred, y = output
        [data_TP, data_FN, data_FP, data_precision, data_recall] = self.cc(y_pred, y)
        self._tp += data_TP
        self._fn += data_FN
        self._fp += data_FP
        self._precision += data_precision
        self._recall += data_recall
        # add all items in current batch
        self._num_examples += 1

    @sync_all_reduce("_precision", "_num_examples")
    def compute(self) -> list():
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        if self._num_examples == 0:
            raise NotComputableError("CC must have at least one example before it can be computed.")
        print("Precision: {}, Num examples: {}, mean precision: {}".format(
            self._precision, self._num_examples, self._precision/self._num_examples))
        return [self._tp / self._num_examples,
                self._fp / self._num_examples,
                self._fn / self._num_examples,
                self._precision / self._num_examples,
                self._recall / self._num_examples]
