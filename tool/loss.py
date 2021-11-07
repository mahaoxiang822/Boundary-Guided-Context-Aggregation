import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import scipy.ndimage as nd
from torch.autograd import Variable


def get_loss(args):
    criterion = JointEdgeSegLoss(classes=args.classes, ignore_index=args.ignore_label, ohem=args.ohem,
                                 aux_weight=args.aux_weight, edge_weight=args.edge_weight, seg_weight=args.seg_weight,
                                 att_weight=args.att_weight).cuda()
    return criterion


def bce2d(input, target):
    n, c, h, w = input.size()
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    ignore_index = (target_t > 1)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0

    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    ignore_index = ignore_index.data.cpu().numpy().astype(bool)

    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num
    weight[ignore_index] = 0
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, ignore_index=255, ohem=False, aux_weight=0.4,
                 edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        self.ignore_index = ignore_index
        if ohem:
            self.seg_loss = OhemCrossEntropy2d(ignore_label=ignore_index)
        else:
            self.seg_loss = ImageBasedCrossEntropyLoss2d(classes=classes, ignore_index=ignore_index, upper_bound=1.0)
        self.aux_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.edge_weight = edge_weight
        self.aux_weight = aux_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input,
                             torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets):
        seg_mask, edge_mask = targets
        losses = {}
        if self.aux_weight > 0:
            seg_in, aux_seg_in, edge_in = inputs
            losses['seg_loss'] = self.seg_weight * self.seg_loss(seg_in, seg_mask) + self.aux_weight * self.aux_loss(
                aux_seg_in, seg_mask)
        else:
            seg_in, edge_in = inputs
            losses['seg_loss'] = self.seg_weight * self.seg_loss(seg_in, seg_mask)
        losses['edge_loss'] = self.edge_weight * 20 * bce2d(edge_in, edge_mask)
        if self.att_weight == 0:
            losses['att_loss'] = 0
        else:
            losses['att_loss'] = self.att_weight * self.edge_attention(seg_in, seg_mask, edge_in)
        return losses


# Img Weighted Loss
class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, reduction="mean", ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                  targets[i].unsqueeze(0))
        return loss


# Cross Entroply NLL Loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss(weight, reduction="mean", ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        # self.min_kept_ratio = float(min_kept_ratio)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor * factor)  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


if __name__ == "__main__":
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    criterion = DistTransLoss2d(num_classes=19)
    inp = torch.rand((2, 19, 64, 64)).cuda().float()
    mask = torch.randint(0, 18, (2, 1, 64, 64)).cuda().long()
    a = time.time()
    loss = criterion(inp, mask)
    print("loss", loss)
    b = time.time()
    loss.backward()
    c = time.time()
    print(b - a)
    print(c - b)
