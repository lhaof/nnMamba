#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation
# from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.nnMamba import nnMambaSeg
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


class nnUNetTrainerV2_MET_mamba(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, split=100):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 200
        self.initial_lr = 1e-4
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.split = split
        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
        #                             len(self.net_num_pool_op_kernel_sizes),
        #                             self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
        #                             dropout_op_kwargs,
        #                             net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
        #                             self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        self.network = nnMambaSeg(in_ch=self.num_input_channels, number_classes=self.num_classes)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")
            splits = []
            splits.append(OrderedDict())
            splits[self.fold]['train'] = np.array(
                ['BraTS-MET-00127-000', 'BraTS-MET-00167-000', 'BraTS-MET-00250-000', 'BraTS-MET-00105-000', 
                'BraTS-MET-00542-000', 'BraTS-MET-00131-000', 'BraTS-MET-00214-000', 'BraTS-MET-00003-000', 
                'BraTS-MET-00543-000', 'BraTS-MET-00227-000', 'BraTS-MET-00108-000', 'BraTS-MET-00223-000', 
                'BraTS-MET-00173-000', 'BraTS-MET-00219-000', 'BraTS-MET-00260-000', 'BraTS-MET-00120-000', 
                'BraTS-MET-00111-000', 'BraTS-MET-00226-000', 'BraTS-MET-00106-000', 'BraTS-MET-00378-000', 
                'BraTS-MET-00233-000', 'BraTS-MET-00300-000', 'BraTS-MET-00129-000', 'BraTS-MET-00380-000', 
                'BraTS-MET-00166-000', 'BraTS-MET-00404-000', 'BraTS-MET-00247-000', 'BraTS-MET-00288-000', 
                'BraTS-MET-00804-000', 'BraTS-MET-00306-000', 'BraTS-MET-00100-000', 'BraTS-MET-00177-000', 
                'BraTS-MET-00264-000', 'BraTS-MET-00538-000', 'BraTS-MET-00294-000', 'BraTS-MET-00016-000', 
                'BraTS-MET-00418-000', 'BraTS-MET-00307-000', 'BraTS-MET-00034-000', 'BraTS-MET-00113-000', 
                'BraTS-MET-00231-000', 'BraTS-MET-00183-000', 'BraTS-MET-00529-000', 'BraTS-MET-00121-000', 
                'BraTS-MET-00410-000', 'BraTS-MET-00230-000', 'BraTS-MET-00033-000', 'BraTS-MET-00036-000', 
                'BraTS-MET-00290-000', 'BraTS-MET-00531-000', 'BraTS-MET-00289-000', 'BraTS-MET-00266-000', 
                'BraTS-MET-00023-000', 'BraTS-MET-00220-000', 'BraTS-MET-00310-000', 'BraTS-MET-00549-000', 
                'BraTS-MET-00133-000', 'BraTS-MET-00349-000', 'BraTS-MET-00412-000', 'BraTS-MET-00008-000', 
                'BraTS-MET-00098-000', 'BraTS-MET-00296-000', 'BraTS-MET-00193-000', 'BraTS-MET-00551-000', 
                'BraTS-MET-00788-000', 'BraTS-MET-00030-000', 'BraTS-MET-00277-000', 'BraTS-MET-00140-000', 
                'BraTS-MET-00282-000', 'BraTS-MET-00537-000', 'BraTS-MET-00283-000', 'BraTS-MET-00006-000', 
                'BraTS-MET-00281-000', 'BraTS-MET-00178-000', 'BraTS-MET-00014-000', 'BraTS-MET-00112-000', 
                'BraTS-MET-00241-000', 'BraTS-MET-00381-000', 'BraTS-MET-00350-000', 'BraTS-MET-00805-000', 
                'BraTS-MET-00408-000', 'BraTS-MET-00406-000', 'BraTS-MET-00262-000', 'BraTS-MET-00545-000', 
                'BraTS-MET-00130-000', 'BraTS-MET-00347-000', 'BraTS-MET-00787-000', 'BraTS-MET-00007-000', 
                'BraTS-MET-00182-000', 'BraTS-MET-00534-000', 'BraTS-MET-00015-000', 'BraTS-MET-00009-000', 
                'BraTS-MET-00116-000', 'BraTS-MET-00211-000', 'BraTS-MET-00025-000', 'BraTS-MET-00286-000', 
                'BraTS-MET-00032-000', 'BraTS-MET-00507-000', 'BraTS-MET-00217-000', 'BraTS-MET-00086-000', 
                'BraTS-MET-00175-000', 'BraTS-MET-00004-000', 'BraTS-MET-00292-000', 'BraTS-MET-00546-000', 
                'BraTS-MET-00104-000', 'BraTS-MET-00807-000', 'BraTS-MET-00797-000', 'BraTS-MET-00375-000', 
                'BraTS-MET-00017-000', 'BraTS-MET-00149-000', 'BraTS-MET-00102-000', 'BraTS-MET-00803-000', 
                'BraTS-MET-00107-000', 'BraTS-MET-00204-000', 'BraTS-MET-00169-000', 'BraTS-MET-00156-000', 
                'BraTS-MET-00794-000', 'BraTS-MET-00125-000', 'BraTS-MET-00119-000', 'BraTS-MET-00810-000', 
                'BraTS-MET-00530-000', 'BraTS-MET-00115-000', 'BraTS-MET-00268-000', 'BraTS-MET-00212-000', 
                'BraTS-MET-00139-000', 'BraTS-MET-00165-000', 'BraTS-MET-00305-000', 'BraTS-MET-00552-000', 
                'BraTS-MET-00122-000', 'BraTS-MET-00132-000', 'BraTS-MET-00202-000', 'BraTS-MET-00415-000', 
                'BraTS-MET-00011-000', 'BraTS-MET-00097-000', 'BraTS-MET-00037-000', 'BraTS-MET-00026-000', 
                'BraTS-MET-00021-000', 'BraTS-MET-00238-000', 'BraTS-MET-00535-000', 'BraTS-MET-00010-000', 
                'BraTS-MET-00274-000', 'BraTS-MET-00541-000'])
            splits[self.fold]['val'] = np.array(
                ['BraTS-MET-00109-000', 'BraTS-MET-00379-000', 'BraTS-MET-00309-000', 'BraTS-MET-00301-000', 
                'BraTS-MET-00808-000', 'BraTS-MET-00304-000', 'BraTS-MET-00411-000', 'BraTS-MET-00136-000', 
                'BraTS-MET-00540-000', 'BraTS-MET-00414-000', 'BraTS-MET-00269-000', 'BraTS-MET-00533-000', 
                'BraTS-MET-00407-000', 'BraTS-MET-00035-000', 'BraTS-MET-00138-000', 'BraTS-MET-00413-000', 
                'BraTS-MET-00791-000', 'BraTS-MET-00031-000', 'BraTS-MET-00243-000', 'BraTS-MET-00799-000', 
                'BraTS-MET-00155-000', 'BraTS-MET-00024-000', 'BraTS-MET-00405-000', 'BraTS-MET-00225-000', 
                'BraTS-MET-00271-000', 'BraTS-MET-00090-000', 'BraTS-MET-00123-000', 'BraTS-MET-00118-000', 
                'BraTS-MET-00280-000', 'BraTS-MET-00110-000', 'BraTS-MET-00020-000', 'BraTS-MET-00159-000', 
                'BraTS-MET-00124-000', 'BraTS-MET-00164-000', 'BraTS-MET-00244-000', 'BraTS-MET-00547-000', 
                'BraTS-MET-00002-000', 'BraTS-MET-00299-000', 'BraTS-MET-00348-000', 'BraTS-MET-00205-000', 
                'BraTS-MET-00027-000', 'BraTS-MET-00005-000', 'BraTS-MET-00272-000', 'BraTS-MET-00019-000', 
                'BraTS-MET-00801-000', 'BraTS-MET-00263-000', 'BraTS-MET-00224-000'])
            split_number = round(len(splits[self.fold]['train']) * (self.split / 100))
            print('split percentage:', self.split / 100)
            save_pickle(splits, splits_file)
            # tr_keys = splits[self.fold]['train']
            tr_keys = splits[self.fold]['train'][:split_number]
            print('number of training cases:', len(tr_keys))
            val_keys = splits[self.fold]['val']
            self.print_to_log_file("This split has %d training and %d validation cases."
                                   % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
