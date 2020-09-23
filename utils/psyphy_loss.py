# Loss function based on Sam's HWR, modified to fit image classification by Jin Huang

import torch
import torch.nn as nn
from torch.nn.modules.loss import CTCLoss



class pp_loss(torch.nn.Module):
    def __init__(self):
        super(pp_loss, self).__init__()
        # self.criterion = CTCLoss(reduction = 'none', zero_infinity=True)
        # self.idx_to_char = idx_to_char
        # self.char_to_idx = char_to_idx
        # self.verbose = verbose
        # Todo: should use cross_entropy?
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, preds, labels, rts):
        """

        :param preds:
        :param labels:
        :param preds_size:
        :param label_lengths:
        :param psych:
        :return:
        """
        loss = self.criterion(preds, labels).cuda()
        index = 0

        label = labels.data.cpu().numpy()
        # TODO: why permute?
        output_batch = preds.permute(1, 0, 2)

        out = output_batch.data.cpu().numpy()
        cer = torch.zeros(loss.shape)

        # TODO: figure out the shape of the loss

        # TODO: adding RT into the loss

        # for j in range(out.shape[0]):
        #     logits = out[j, ...]
        #     # Change this part...
        #     pred, raw_pred = string_utils.naive_decode(logits)
        #     pred_str = string_utils.label2str(pred, self.idx_to_char, False)
        #     gt_str = string_utils.label2str(lbl[index:lbl_len[j] + index], self.idx_to_char, False)
        #     index += lbl_len[j]
        #     cer[j] = error_rates.cer(gt_str, pred_str)

        cer = Variable(cer, requires_grad = True).cuda()
        loss = loss + (psych * cer)

        return torch.sum(loss)