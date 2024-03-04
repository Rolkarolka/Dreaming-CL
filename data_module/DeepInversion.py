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

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import os
import glob
import collections
import matplotlib

from torch.utils.data import TensorDataset
from torchvision.models import resnet18


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

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
    def __init__(
            self,
            class_num_samples,
            logger,
            debug_output=True,
            epochs=1000,
            di_lr=0.1,
            competitive_scale=0.0,
            di_l2_scale=3e-8,
            di_r_feature=100.0,  #{1:0; 5:0; 10:0; 100:0} TODO adaptive learning rate
            di_main_loss_multiplier = 1.0,
            batch_size=64,
            tv_l1 = 0.0,
            tv_l2 = 0.0001
    ):
        self.di_lr = di_lr
        self.logger = logger
        self.epochs = epochs
        self.debug_output = debug_output
        self.competitive_scale = competitive_scale
        self.di_l2_scale = di_l2_scale
        self.di_r_feature = di_r_feature
        self.di_main_loss_multiplier = di_main_loss_multiplier
        self.class_num_samples = class_num_samples
        self.batch_size = batch_size
        self.tv_l1 = tv_l1
        self.tv_l2 = tv_l2

        self.jitter = 30
        self.do_flip = True
        self.first_bn_multiplier = 10.


    def _get_images(self, net, device, targets, inputs, num_classes,
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
            optimizer: optimizer to be used for model inversion
            inputs: data placeholder for optimization, will be reinitialized to noise
            bn_reg_scale: weight for r_feature_regularization
            random_labels: sample labels from random distribution or use columns of the same class
            l2_coeff: coefficient for L2 loss on input
        return:
            A tensor on GPU with shape (batch_size, 3, 32, 32) for CIFAR
        '''

        kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
        best_cost = 1e6
        criterion = nn.CrossEntropyLoss()

        net_student.eval()

        optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        ## Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        # setting up the range for jitter
        lim_0, lim_1 = 2, 2

        skipfirst = False
        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 1000 if not skipfirst else 2000

            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            optimizer = optim.Adam([inputs], lr=self.di_lr, betas=[0.5, 0.9], eps=1e-8)
            do_clip = True
            lr_scheduler = lr_cosine_policy(self.di_lr, 100, iterations_per_layer)


            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)
                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))


                # foward with jit images
                optimizer.zero_grad()
                net.zero_grad()

                outputs = net(inputs_jit)

                # R_cross classification loss
                loss = criterion(outputs, targets)

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # R_feature loss
                rescale = [self.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                loss_r_feature = sum(
                    [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                # competition loss, Adaptive DeepInvesrion
                loss_verifier_cig = torch.zeros(1)
                if self.competitive_scale != 0.0:
                    net_student.zero_grad()
                    outputs_student = net_student(inputs_jit)
                    T = 3.0

                    # jensen shanon divergence:
                    # another way to force KL between negative probabilities
                    Q = F.softmax(outputs / T, dim=1)
                    P = F.softmax(outputs_student / T, dim=1)
                    P = P[:,:num_classes]
                    M = 0.5 * (P + Q)

                    P = torch.clamp(P, 0.01, 0.99)
                    Q = torch.clamp(Q, 0.01, 0.99)
                    M = torch.clamp(M, 0.01, 0.99)
                    eps = 0.0
                    loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                    # JS criteria - 0 means full correlation, 1 - means completely different
                    loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                    # loss = loss + self.competitive_scale * loss_verifier_cig

                # l2 loss
                loss_l2 = torch.norm(inputs_jit.view(self.batch_size, -1), dim=1).mean()

                loss_aux = self.tv_l2 * loss_var_l2 + \
                           self.tv_l1 * loss_var_l1 + \
                           self.di_r_feature * loss_r_feature + \
                           self.di_l2_scale * loss_l2

                if self.competitive_scale != 0.0:
                    loss_aux += self.competitive_scale * loss_verifier_cig

                loss = self.di_main_loss_multiplier * loss + loss_aux
                loss.backward()
                optimizer.step()

                if self.debug_output and iteration % 200 == 0:
                    print("------------iteration {}----------".format(iteration))
                    print("total loss", loss.item())
                    print("loss_r_feature", loss_r_feature.item())
                    print("main criterion", criterion(outputs, targets).item())

                    # vutils.save_image(inputs.data.clone(),
                    #                   './{}/output_{}.png'.format(prefix, epoch // 200),
                    #                   normalize=True, scale_each=True, nrow=10)

                if do_clip:
                    inputs.data = clip(inputs.data)

                if best_cost > loss.item():
                    best_cost = loss.item()
                    best_inputs = inputs.data

        outputs = net(best_inputs)
        _, predicted_teach = outputs.max(1)

        outputs_student = net_student(best_inputs)
        _, predicted_std = outputs_student.max(1)

        print('Teacher correct out of {}: {}, loss at {}'.format(self.batch_size,
                                                                 predicted_teach.eq(targets).sum().item(),
                                                                 criterion(outputs, targets).item()))
        print(
            'Student correct out of {}: {}, loss at {}'.format(self.batch_size, predicted_std.eq(targets).sum().item(),
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

    def run_inversion(self, net_teacher, net_student, classes_to_dream):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        net_student = net_student.to(device)
        net_teacher = net_teacher.to(device)

        net_teacher.eval()  # important, otherwise generated images will be non natural
        cudnn.benchmark = True

        prefix = "runs/data_generation/" + self.logger.run_id + "/"

        for create_folder in [prefix, prefix + "/best_images/"]:
            if not os.path.exists(create_folder):
                os.makedirs(create_folder)

        # placeholder for inputs
        data_type = torch.float
        all_probes = sum(list(self.class_num_samples.values()))
        all_targets = []
        for key, value in self.class_num_samples.items():
            all_targets += [key for _ in range(value)]
        random.shuffle(all_targets)

        dreamed_inputs = torch.Tensor()
        dreamed_targets = torch.LongTensor()
        i = 0
        while dreamed_targets.numel() < all_probes:
            inputs = torch.randn((self.batch_size, 3, 32, 32), requires_grad=True, device=device, dtype=data_type)
            targets = torch.LongTensor(all_targets[i * self.batch_size: i * self.batch_size + self.batch_size]).to(
                device)
            optimizer_di = optim.Adam([inputs], lr=self.di_lr)

            print(f"Starting {i}/{all_probes // self.batch_size} model inversion")
            inputs = self._get_images(net=net_teacher, targets=targets,
                                      net_student=net_student, prefix=prefix,
                                      optimizer=optimizer_di, inputs=inputs,
                                      device=device, num_classes=len(classes_to_dream))

            dreamed_targets = torch.cat([dreamed_targets, targets.detach().cpu()], dim=0)
            dreamed_inputs = torch.cat([dreamed_inputs, inputs.detach().cpu()], dim=0)
            i += 1

        trained_grid_path = os.path.join(os.getcwd(), 'trained')
        if not os.path.isdir(trained_grid_path):
            os.makedirs(trained_grid_path)

        num_class_probs = 5
        for class_name in classes_to_dream:
            class_indices = torch.nonzero(dreamed_targets == class_name).squeeze()
            random_indices = random.sample(class_indices.tolist(), min(num_class_probs, len(class_indices)))
            grid = vutils.make_grid(dreamed_inputs[random_indices], normalize=True, scale_each=True, nrow=5)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            img_path = os.path.join(trained_grid_path, f"dreamed_class_target_{class_name}.png")
            matplotlib.image.imsave(img_path, ndarr)
            self.logger.experiment.log_artifact(self.logger.run_id,  img_path)

        dataset = TensorDataset(dreamed_inputs, dreamed_targets)
        return dataset, net_student
