#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import OrderedDict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import collections

class _BaseWrapper(object):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        # simple classification
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """
        Class-specific backpropagation

        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """
        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_total = OrderedDict()
        self.grad_total = OrderedDict()
        self.fmap_pool = collections.defaultdict(dict)
        self.grad_pool = collections.defaultdict(dict)
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps for layers with parameters
                if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                    self.fmap_total[key] = output
                else:
                    if len(module._parameters) > 0:
                        self.fmap_pool[key][module.weight.device] = output#.detach()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps for layers with parameters
                if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                    self.grad_total[key] = grad_out[0]
                else:
                    if len(module._parameters) > 0:
                        self.grad_pool[key][module.weight.device] = grad_out[0]#.detach()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image):
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image)

    def generate(self, target_layer, per_pixel=False):
        #if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        #    self.fmap_total = OrderedDict()
        #    for key, val in self.fmap_pool.items():
        #        assert(len(list(val.values())) == 1)
        #        self.fmap_total[key] = list(val.values())[0]
        #    self.grad_total = OrderedDict()
        #    for key, val in self.grad_pool.items():
        #        assert(len(list(val.values())) == 1)
        #        self.grad_total[key] = list(val.values())[0]
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.fmap_total = self.aggregate_items(self.fmap_pool)
            self.grad_total = self.aggregate_items(self.grad_pool)
        fmaps = self._find(self.fmap_total, target_layer)
        grads = self._find(self.grad_total, target_layer)
        if per_pixel:
            weights = grads
        else:
            weights = self._compute_grad_weights(grads)

        weights = weights.to(fmaps.device) # put on same GPU if necessary
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam_max = gcam.max(dim=1, keepdim=True)[0]
        gcam_max[torch.where(gcam_max == 0)] = 1. # prevent divide by 0
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        #gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam /= gcam_max
        gcam = gcam.view(B, C, H, W)

        return gcam

    def aggregate_items(self, nested):
        """
        Input: nested dictionary with 2 levels;
                 - first set of keys is key in forward & backward hooks (module names)
                 - second set of keys is torch.device object that called hook.
               Items are either the feature map activations (for forward hook)
               or gradients (for backward hook).

        Returns: single level dictionary, where keys are identical to
                 first set of keys in nested, and each item
                 output[key] = concatenation(all of nested[key] tensors)
                 Concatenation occurs in order of GPU device IDs, if using cuda
                 output[key] first dimension is now equal to batch size
        """
        output = OrderedDict()
        sort_gpu = lambda x: x[0].index
        for key in nested.keys():
            list_items = list(nested[key].items())
            if torch.cuda.is_available():
                # Sort list based on GPU device ID
                list_items = sorted(list_items, key=sort_gpu)
                # Get out tensors and move all to first GPU
                gpu_0 = list_items[0][0]
                tensors = [i[1] for i in list_items]
                for i in range(len(tensors)):
                    tensors[i] = tensors[i].to(gpu_0)
                
                if len(tensors) > 1: 
                    all_tensors = torch.cat(tensors, axis=0)
                else:
                    all_tensors = tensors[0]
                output[key] = all_tensors
            else:
                # Only CPU as second set of keys
                assert(len(y[key].keys()) == 1)
                output[key] = nested[key][0]
        return output


