'''
ResNet model inversion for CIFAR10.

Copyright (C) 2020 NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License (1-Way Commercial). To view a copy of this license,
visit https://github.com/NVlabs/DeepInversion/blob/master/LICENSE
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import os
import glob
import collections

from torch.utils.data import TensorDataset
from torchvision.models import resnet18

class DeepInversionFeatureHook:
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class DeepInversion:
    def __init__(self, batch_size=256, debug_output=False):
        self.di_lr = 0.1
        self.epochs = 2000
        self.debug_output = debug_output
        self.competitive_scale = 0.0
        self.di_var_scale = 2.5e-5
        self.di_l2_scale = 0.0
        self.r_feature_weight = 1e2
        self.batch_size = batch_size
        self.exp_descr = "try1"

    def _get_images(self, net, device, targets, inputs,
                    net_student=None, prefix=None,
                    optimizer=None):
        '''
        Function returns inverted images from the pretrained model, parameters are tight to CIFAR dataset
        args in:
            net: network to be inverted
            batch_size: batch size
            epochs: total number of iterations to generate inverted images, training longer helps a lot!
            idx: an external flag for printing purposes: only print in the first round, set as -1 to disable
            var_scale: the scaling factor for variance loss regularization. this may vary depending on batch_size
                larger - more blurred but less noise
            net_student: model to be used for Adaptive DeepInversion
            prefix: defines the path to store images
            competitive_scale: coefficient for Adaptive DeepInversion
            train_writer: tensorboardX object to store intermediate losses
            global_iteration: indexer to be used for tensorboard
            use_amp: boolean to indicate usage of APEX AMP for FP16 calculations - twice faster and less memory on TensorCores
            optimizer: potimizer to be used for model inversion
            inputs: data place holder for optimization, will be reinitialized to noise
            bn_reg_scale: weight for r_feature_regularization
            random_labels: sample labels from random distribution or use columns of the same class
            l2_coeff: coefficient for L2 loss on input
        return:
            A tensor on GPU with shape (batch_size, 3, 32, 32) for CIFAR
        '''

        kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)

        # preventing backpropagation through student for Adaptive DeepInversion
        net_student.eval()

        best_cost = 1e6

        # initialize gaussian inputs
        inputs.data = torch.randn((self.batch_size, 3, 32, 32), requires_grad=True, device=device)
        # if use_amp:
        #     inputs.data = inputs.data.half()

        # set up criteria for optimization
        criterion = nn.CrossEntropyLoss()

        optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer

        ## Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        # setting up the range for jitter
        lim_0, lim_1 = 2, 2

        for epoch in range(self.epochs):
            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            # foward with jit images
            optimizer.zero_grad()
            net.zero_grad()
            outputs = net(inputs_jit)
            loss = criterion(outputs, targets)
            loss_target = loss.item()

            # competition loss, Adaptive DeepInvesrion
            if self.competitive_scale != 0.0:
                net_student.zero_grad()
                outputs_student = net_student(inputs_jit)
                T = 3.0

                if 1:
                    # jensen shanon divergence:
                    # another way to force KL between negative probabilities
                    P = F.softmax(outputs_student / T, dim=1)
                    Q = F.softmax(outputs / T, dim=1)
                    M = 0.5 * (P + Q)

                    P = torch.clamp(P, 0.01, 0.99)
                    Q = torch.clamp(Q, 0.01, 0.99)
                    M = torch.clamp(M, 0.01, 0.99)
                    eps = 0.0
                    # loss_verifier_cig = 0.5 * kl_loss(F.log_softmax(outputs_verifier / T, dim=1), M) +  0.5 * kl_loss(F.log_softmax(outputs/T, dim=1), M)
                    loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                    # JS criteria - 0 means full correlation, 1 - means completely different
                    loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                    loss = loss + self.competitive_scale * loss_verifier_cig

            # apply total variation regularization
            diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
            diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
            diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
            diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss = loss + self.di_var_scale * loss_var

            # R_feature loss
            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
            loss = loss + self.r_feature_weight * loss_distr  # best for noise before BN

            # l2 loss
            if 1:
                loss = loss + self.di_l2_scale * torch.norm(inputs_jit, 2)

            if self.debug_output and epoch % 200 == 0:
                print(
                    f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")
                vutils.save_image(inputs.data.clone(),
                                  './{}/output_{}.png'.format(prefix, epoch // 200),
                                  normalize=True, scale_each=True, nrow=10)

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

            # backward pass
            loss.backward()

            optimizer.step()

        outputs = net(best_inputs)
        _, predicted_teach = outputs.max(1)

        outputs_student = net_student(best_inputs)
        _, predicted_std = outputs_student.max(1)

        print('Teacher correct out of {}: {}, loss at {}'.format(self.batch_size, predicted_teach.eq(targets).sum().item(),
                                                                 criterion(outputs, targets).item()))
        print('Student correct out of {}: {}, loss at {}'.format(self.batch_size, predicted_std.eq(targets).sum().item(),
                                                                 criterion(outputs_student, targets).item()))

        name_use = "best_images"
        if prefix is not None:
            name_use = prefix + name_use
        next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1

        vutils.save_image(best_inputs[:20].clone(),
                          './{}/output_{}.png'.format(name_use, next_batch),
                          normalize=True, scale_each=True, nrow=10)

        net_student.train()

        return best_inputs

    def run_inversion(self, net_teacher, classes_to_dream):
        net_student = resnet18()
        student_num_features = net_student.fc.in_features
        net_student.fc = nn.Linear(student_num_features, len(classes_to_dream))

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        net_student = net_student.to(device)
        net_teacher = net_teacher.to(device)

        # placeholder for inputs
        data_type = torch.float
        inputs = torch.randn((self.batch_size, 3, 32, 32), requires_grad=True, device=device, dtype=data_type)
        targets = torch.LongTensor([random.choice(classes_to_dream) for _ in range(self.batch_size)]).to(device)

        optimizer_di = optim.Adam([inputs], lr=self.di_lr)

        net_teacher.eval()  # important, otherwise generated images will be non natural
        cudnn.benchmark = True

        prefix = "runs/data_generation/" + self.exp_descr + "/"

        for create_folder in [prefix, prefix + "/best_images/"]:
            if not os.path.exists(create_folder):
                os.makedirs(create_folder)

        print("Starting model inversion")

        inputs = self._get_images(net=net_teacher, targets=targets,
                                  net_student=net_student, prefix=prefix,
                                  optimizer=optimizer_di, inputs=inputs,
                                  device=device)

        dataset = TensorDataset(inputs, targets)
        return dataset
