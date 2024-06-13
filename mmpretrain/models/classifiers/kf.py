import copy
import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn
from mmpretrain.models.losses.kd_loss import InfoMax_loss, InfoMin_loss
from mmpretrain.models.builder import build_backbone, build_neck, build_loss, build_head
from mmpretrain.models.utils.augment import Augments

from .base import BaseClassifier
from mmpretrain.registry import MODELS
from typing import List, Optional
from mmpretrain.structures import DataSample



@MODELS.register_module()
class KFImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone: dict,
                 kd_loss: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 aug_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 ):
        
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'ClsDataPreprocessor')
            data_preprocessor.setdefault('batch_augments', aug_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super(KFImageClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        assert 'student' in backbone.keys(), 'student network should be specified'
        assert 'teacher' in backbone.keys(), 'teacher network should be specified'
        return_tuple = backbone.pop('return_tuple', True)

        self.num_task = backbone["num_task"]
        self.student = nn.ModuleDict(
            {
                "CKN": build_backbone(backbone["student"]["CKN"]),
                "TSN": nn.ModuleList([build_backbone(backbone["student"]["TSN"]) for i in range(self.num_task)]),
                "neck": build_neck(neck["student"]),
                "head_task": build_head(head["task"]),
                "head": build_head((head["student"]))  # CKN的分类头
            }
        )
        self.teacher = nn.ModuleDict(
            {
                "backbone": build_backbone(backbone["teacher"]),
                "neck": build_neck(neck["teacher"]),
                "head": build_head(head["teacher"]),
            }
        )

        # TODO: 为什么要重新加一个映射层？
        self.feat_channels_student = train_cfg['feat_channels']['student']  # [128, 256, 512]
        self.feat_channels_teacher = train_cfg['feat_channels']['teacher']  # [512, 1024, 2048]
        feat_fcs = []
        for i in range(len(self.feat_channels_student)):
            feat_fcs.append(nn.Sequential(
                nn.Linear(
                    self.feat_channels_teacher[i], self.feat_channels_student[i]),
                nn.BatchNorm1d(self.feat_channels_student[i]),
                nn.ReLU(True),
                # nn.Linear(
                #     self.feat_channels_student[i], self.feat_channels_student[i])
            )
            )
        self.feat_fcs = nn.ModuleList(feat_fcs)

        # self.criterionCls = F.cross_entropy
        # self.criterionTask = F.binary_cross_entropy_with_logits
        self.criterionKD = build_loss(kd_loss)

        self.lambda_kd = train_cfg["lambda_kd"]   # kd para
        self.lambda_feat = train_cfg["lambda_feat"]  # info_max para
        self.alpha = train_cfg["alpha"]
        self.beta = train_cfg["beta"]  # info_min para
        self.teacher_ckpt = train_cfg["teacher_checkpoint"]
        self.task_weight = train_cfg["task_weight"]
        self.return_tuple = return_tuple

        self.load_teacher()  # 加载teacher模型权重

        # self.augments = None
        # if train_cfg is not None:
        #     augments_cfg = train_cfg["augments", None]
        #     if augments_cfg is not None:
        #         self.augments = Augments(augments_cfg)

    def load_teacher(self):
        split_lins = '*' * 20
        state_dict = torch.load(self.teacher_ckpt)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        try:
            self.teacher.load_state_dict(state_dict)
            print(split_lins)
            print(
                f'Teacher pretrained model has been loaded {self.teacher_ckpt}')
            print(split_lins)
        except:
            print('Teacher model not loaded')
            print(state_dict.keys())
            print(self.teacher.state_dict().keys())
            AssertionError('Teacher model not loaded')
            exit()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def extract_feat(self, inputs):
        pass

    # functions for teacher network
    def extract_teacher_feat(self, inputs):
        """ backbone + neck"""
        x = self.teacher["backbone"](inputs)
        if self.return_tuple:
            if not isinstance(x, tuple):
                x = (x,)
        else:
            if isinstance(x, tuple):
                x = x[-1]
        x = self.teacher["neck"](x)
        return x

    def get_teacher_logit(self, inputs):
        x = self.extract_teacher_feat(inputs)
        if isinstance(x, tuple):
            last_x = x[-1]
        logit = self.teacher["head"].fc(last_x)
        return logit, x

    # functions for student network
    def extract_CKN_feat(self, inputs):
        x = self.student["CKN"](inputs)       
        if self.return_tuple:
            if not isinstance(x, tuple):
                x = (x,)
        else:
            if isinstance(x, tuple):
                x = x[-1]
        x = self.student["neck"](x)
        return x

    def extract_TSN_feat(self, inputs):
        result = dict(feats=[], mu_vars=[])
        for i in range(self.num_task):
            (mu, var), x = self.student["TSN"][i](inputs)
            if self.return_tuple:
                if not isinstance(x, tuple):
                    x = (x,)
            else:
                if isinstance(x, tuple):
                    x = x[-1]
            result["feats"].append(x)
            result["mu_vars"].append((mu, var))
        return result

    def extract_student_feats(self, inputs):
        CKN_result = self.extract_CKN_feat(inputs)
        TSN_result = self.extract_TSN_feat(inputs)

        if self.num_task == 1:
            return CKN_result, TSN_result["feats"][0], TSN_result
        else:
            return CKN_result, TSN_result["feats"], TSN_result

    def get_student_logit(self, inputs):
        # CKN_result feature len为3
        # CKN_result1是最后1个
        CKN_result, TSN_feat, TSN_result = self.extract_student_feats(inputs)
        if isinstance(CKN_result, tuple):
            CKN_result1 = CKN_result[-1]
        if isinstance(TSN_feat, tuple):
            TSN_feat = TSN_feat[-1]
        if isinstance(TSN_feat, list):
            feat = [CKN_result1 + TSN_subfeat[-1] for TSN_subfeat in TSN_feat]
        else:
            feat = CKN_result1 + TSN_feat
        logit = self.student["head"].get_logits(feat)
        TSN_logit = self.student['head_task'].get_logits(TSN_feat)
        return logit, TSN_logit, CKN_result, TSN_result

    def get_logit(self, inputs):
        return self.get_student_logit(inputs)[0]

    def get_adv_logit(self, inputs):
        return self.get_student_logit(inputs)[1]


    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:

        with torch.no_grad():
            teacher_logit, teacher_feature = self.get_teacher_logit(inputs)

        student_logit, TSN_logit, CKN_result, TSN_result = self.get_student_logit(inputs)
 
        # INFOMax loss
        loss_infomax = 0.
        assert len(teacher_feature) == len(CKN_result)
        for layer_id, (teacher_x_layer, student_x_layer) in enumerate(zip(teacher_feature, CKN_result)):
            loss_infomax += InfoMax_loss(self.feat_fcs[layer_id](teacher_x_layer), student_x_layer) * self.lambda_feat
        loss_infomax = loss_infomax / len(CKN_result)

        # KD loss
        loss_kd = self.criterionKD(student_logit, teacher_logit.detach()) * self.lambda_kd

        # cls loss (CKN的cls)
        loss_cls = self.student["head"].loss(student_logit, data_samples)

        # TSN loss（TSN的cls)
        # TODO: 这个到底算什么？
        loss_task = self.student["head_task"].loss(TSN_logit, data_samples) * self.task_weight

        # InfoMin loss
        loss_infomin = 0.
        for mu, log_var in TSN_result["mu_vars"]:
            loss_infomin += InfoMin_loss(mu, log_var) * self.beta
        

        losses = dict(loss_infomax=loss_infomax,
                      loss_kd=loss_kd,
                      loss_cls=loss_cls,
                      loss_task=loss_task,
                      loss_infomin=loss_infomin)
        return losses

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'loss'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor(s) without any
          post-processing, same as a common PyTorch Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """

        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
        
    # # TODO:要更新predict，head也要写predict
    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        cls_score= self.get_logit(inputs)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        pred_labels = pred.argmax(dim=1, keepdim=True).detach()
        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred.size(0))]

        for data_sample, score, label in zip(data_samples, pred,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        
        return out_data_samples
    
    
    # def forward_train(self, input, gt_label, **kwargs):
    #     """Forward computation during training.

    #     Args:
    #         img (Tensor): of shape (N, C, H, W) encoding input images.
    #             Typically these should be mean centered and std scaled.
    #         gt_label (Tensor): It should be of shape (N, 1) encoding the
    #             ground-truth label of input images for single label task. It
    #             shoulf be of shape (N, C) encoding the ground-truth label
    #             of input images for multi-labels task.
    #     Returns:
    #         dict[str, Tensor]: a dictionary of loss components
    #     """

    #     if self.augments is not None:
    #         input, gt_label = self.augments(input, gt_label)

    #     with torch.no_grad():
    #         teacher_logit, teacher_feature = self.get_teacher_logit(input)
    #         student_logit, TSN_logit, CKN_result, TSN_result = self.get_student_logit(input)

    #     # INFOMax loss
    #     loss_infomax = 0.
    #     assert len(teacher_feature) == len(CKN_result)
    #     for layer_id, (teacher_x_layer, student_x_layer) in enumerate(zip(teacher_feature, CKN_result)):
    #         loss_infomax += InfoMax_loss(self.feat_fcs[layer_id](teacher_x_layer), student_x_layer) * self.lambda_feat
    #     loss_infomax = loss_infomax / len(CKN_result)

    #     # KD loss
    #     loss_kd = self.criterionKD(student_logit, teacher_logit.detach()) * self.lambda_kd

    #     # cls loss
    #     loss_cls = self.student["head"].loss(student_logit, gt_label)["loss"]

    #     # TSN loss
    #     # TODO: 这个到底算什么？
    #     loss_task = self.head["head_task"].loss(TSN_logit, gt_label)["loss"] * self.task_weight

    #     # InfoMin loss
    #     loss_infomin = 0.
    #     for mu, log_var in TSN_result["mu_vars"]:
    #         loss_infomin += InfoMin_loss(mu, log_var) * self.beta
    #     losses = dict(loss_infomax=loss_infomax,
    #                   loss_kd=loss_kd,
    #                   loss_cls=loss_cls,
    #                   loss_task=loss_task,
    #                   loss_infomin=loss_infomin)
    #     return losses
