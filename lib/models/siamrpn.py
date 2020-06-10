# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Reference: SiamRPN [Li]
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SiamRPN_(nn.Module):
    def __init__(self, anchors_nums=None, cls_type='thicker'):
        """
        :param cls_loss: thinner or thicker
                        thinner: output [B, 5, 17, 17] with BCE loss
                        thicker: output [B, 10, 17, 17] with nll loss
        """
        super(SiamRPN_, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None  # for online tracking
        self.anchor_nums = anchors_nums
        self.cls_type = cls_type


        self.criterion = nn.BCEWithLogitsLoss() 

        #the first loss function we've implemented
        #self.criterion = nn.HingeEmbeddingLoss()

        #focal loss is implemented but not evaluated - multiple versions of Focall loss is 
        #provided at the end of this code
        
        #self.criterion = FocalLoss()


    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_cls, pred_reg = self.connect_model(template_feature, search_feature)
        return pred_cls, pred_reg

    def template(self, z):
        self.zf = self.feature_extractor(z)

    def track(self, x):
        xf = self.feature_extractor(x)
        pred_cls, pred_reg = self.connector(self.zf, xf)
        return pred_cls, pred_reg

    # ------- For Training ---------
    def _weight_l1_loss(self, pred_reg, label_reg, weight):
        """
        for reg loss (smooth l1 also works)
        """
        b, _, sh, sw = pred_reg.size()
        pred_reg = pred_reg.view(b, 4, -1, sh, sw)
        diff = (pred_reg - label_reg).abs()
        diff = diff.sum(dim=1).view(b, -1, sh, sw)
        loss = diff * weight
        return loss.sum().div(b)

    # cls loss thinner--
    def _get_loss(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)   # the same as tf version


    def _cls_loss(self, pred, label, posFLAG=True):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
        neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()
        loss_neg = self._get_loss(pred, label, neg)
        if not posFLAG:
            return loss_neg
        else:
            loss_pos = self._get_loss(pred, label, pos)
            return loss_pos * 0.5 + loss_neg * 0.5

    def _loss(self, label_cls, label_reg, reg_weight, pred_cls, pred_reg, sum_reg):
        """
        cls loss and reg loss
        """
        b, a2, h, w = pred_cls.size()
        cls_sum = torch.sum(pred_cls, 1)  # sum reg
        cls_loss = self._cls_loss(pred_cls, label_cls) + 0.1 * self._cls_loss(cls_sum, sum_reg, posFLAG=False)
        reg_loss = self._weight_l1_loss(pred_reg, label_reg, reg_weight)

        return cls_loss, reg_loss

    # cls loss thicker--
    def _loss_thicker(self, label_cls, label_reg, reg_weight, pred_cls, pred_reg):
        """
        cls loss and reg loss
        """
        b, c, h, w = pred_cls.size()
        pred_cls = pred_cls.view(b, 2, c // 2, h, w)
        pred_cls = pred_cls.permute(0, 2, 3, 4, 1).contiguous()
        pred_cls = F.log_softmax(pred_cls, dim=4)
        pred_cls = pred_cls.contiguous().view(-1, 2)

        cls_loss = self._weighted_CE(pred_cls, label_cls)
        reg_loss = self._weight_l1_loss(pred_reg, label_reg, reg_weight)
        return cls_loss, reg_loss


    def _weighted_CE(self, pred, label):
        """
        for cls loss
        label_cls  -- 1: positive, 0: negative, -1: ignore
        """
        pred = pred.view(-1, 2)
        label = label.view(-1)
        pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
        neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()
			
        loss_pos = self._cls_loss_thicker(pred, label, pos)
        loss_neg = self._cls_loss_thicker(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def _cls_loss_thicker(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)

        return F.nll_loss(pred, label)

    def forward(self, template, search, label_cls=None, label_reg=None, reg_weight=None, sum_weight=None):
        zf = self.feature_extractor(template)
        xf = self.feature_extractor(search)
        pred_cls, pred_reg = self.connector(zf, xf)
        if self.training:

            if self.cls_type == 'thinner':
                cls_loss, reg_loss = self._loss(label_cls, label_reg, reg_weight, pred_cls, pred_reg, sum_weight)
                print("reg loss eddited")
            elif self.cls_type == 'thicker':
                cls_loss, reg_loss = self._loss_thicker(label_cls, label_reg, reg_weight, pred_cls, pred_reg)
                print("reg loss eddited 2")
            else:
                raise ValueError('not implemented loss type')
            return cls_loss, reg_loss
        else:
            raise ValueError('forward is only used for training.')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
           BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') + 1e-6
           #print("BCE_LOSS ------------------------------", BCE_loss)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none') + 1e-6
            #print("BCE_LOSS ------------------------------", BCE_loss)
        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * ((1-pt)**self.gamma + 1e-6) * (BCE_loss + 1e-6)

        if self.reduction:
            return torch.mean(F_loss+1e-6) + 1e-6
        else:
            return F_loss + 1e-6

class FocalLoss2(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        size = target.size() + (input.size(-1),)
        view = target.size() + (1,)

        mask = torch.Tensor(*size).fill_(0).cuda()
        target = target.view(*view).cuda()
        ones = 1.

        if isinstance(target, Variable):
            ones = Variable(torch.Tensor(target.size()).fill_(1))
            mask = Variable(mask, volatile=target.volatile)

        y = mask.scatter_(1, target, ones).cuda()

        logit = F.softmax(input, dim=-1).cuda()
        logit = logit.clamp(self.eps, 1. - self.eps).cuda()
        loss = -1 * y * torch.log(logit).cuda() # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum().cuda()

class FocalLoss3(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss3, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): 
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
