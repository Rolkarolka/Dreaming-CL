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
import matplotlib

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
    def __init__(
            self,
            class_num_samples,
            logger,
            debug_output=True,
            epochs=1000,
            di_lr=0.1,
            competitive_scale=0.0,
            di_var_scale=0.001,
            di_l2_scale=3e-8,
            di_r_feature=100.0,  #{1:0; 5:0; 10:0; 100:0} TODO adaptive learning rate
            batch_size=64,
    ):
        self.di_lr = di_lr
        self.logger = logger
        self.epochs = epochs
        self.debug_output = debug_output
        self.competitive_scale = competitive_scale
        self.di_var_scale = di_var_scale
        self.di_l2_scale = di_l2_scale
        self.di_r_feature = di_r_feature
        self.class_num_samples = class_num_samples
        self.batch_size = batch_size

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

        # preventing backpropagation through student for Adaptive DeepInversion
        net_student.eval()

        best_cost = 1e6
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
            loss = loss + self.di_r_feature * loss_distr  # best for noise before BN

            # l2 loss
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
