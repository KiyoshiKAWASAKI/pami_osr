"""Psychophysics losses in Pytorch"""

import torch

class PsychophysicsLoss(torch.nn.
    """Loss for classification using psychophysics."""
    def __init__(self, loss_id, psychophysics_stats):
        loss_id = loss_id.lower()

        if loss_id in ['norm_diff_to_overall_max_rt', 'grieggs']:
            return
        elif loss_id in ['norm_diff_to_overall_max_rt', 'grieggs']:
        elif loss_id in ['cumulative_gaussian']:

        self.loss_id = loss_id
        self.psychophysics_stats = psychophysics_stats

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return

# Think cumulative distrib function, so it is monotonically increasing

# TODO normalized difference to image max
class norm_diff_to_max_rt(input, max_rt, scale=1, bias=1):
    """The normalized difference of the instance's reaction time to the max
    overall reaction time.

    `(max_reaction_time - instance_reaction_time) / max_reaction_time`

    Args
    ----
    scale : float
        The value used to scale the resulting normalized difference.
    bias : float
        The value to add to the resulting normalized difference after scaling.
    max_rt : float
        The maximum reaction time
    """
    return torch.nn.functional.linear((max_rt - input)/max_rt, scale, bias)

#   TODO normalized difference to image mean, if greater
#   TODO normalized difference to image median, if greater

# TODO normalized difference to class max

# TODO normalized difference to class max
