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
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation

# from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.nnMamba2 import nnMambaSeg

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


class nnUNetTrainerV2_GIL_mamba2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, split=100):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-4
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        self.split = split

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

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            splits = load_pickle(splits_file)

            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
            else:
                self.print_to_log_file("INFO: Requested fold %d but split file only has %d folds. I am now creating a "
                                       "random 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

        # if self.fold == "all":
        #     # if fold==all then we use all images for training and validation
        #     tr_keys = val_keys = list(self.dataset.keys())
        # else:
        #     splits_file = join(self.dataset_directory, "splits_final.pkl")
        #     splits = []
        #     splits.append(OrderedDict())
        #     splits[self.fold]['train'] = np.array(
        #         ['BraTS-GLI-00580-000', 'BraTS-GLI-01162-000', 'BraTS-GLI-00683-000', 'BraTS-GLI-01364-000', 'BraTS-GLI-01070-000', 'BraTS-GLI-00022-000', 
        #         'BraTS-GLI-01268-000', 'BraTS-GLI-01035-000', 'BraTS-GLI-01195-000', 'BraTS-GLI-01376-000', 'BraTS-GLI-01144-000', 'BraTS-GLI-01174-000', 
        #         'BraTS-GLI-00096-000', 'BraTS-GLI-00113-000', 'BraTS-GLI-01259-000', 'BraTS-GLI-00344-000', 'BraTS-GLI-00674-001', 'BraTS-GLI-00582-000', 
        #         'BraTS-GLI-01241-000', 'BraTS-GLI-00207-000', 'BraTS-GLI-00410-000', 'BraTS-GLI-00545-000', 'BraTS-GLI-00548-001', 'BraTS-GLI-00608-001', 
        #         'BraTS-GLI-00727-001', 'BraTS-GLI-01134-000', 'BraTS-GLI-00520-000', 'BraTS-GLI-01078-000', 'BraTS-GLI-00802-000', 'BraTS-GLI-01123-000', 
        #         'BraTS-GLI-01456-000', 'BraTS-GLI-01121-000', 'BraTS-GLI-01096-000', 'BraTS-GLI-01252-000', 'BraTS-GLI-01474-000', 'BraTS-GLI-01082-000', 
        #         'BraTS-GLI-00656-000', 'BraTS-GLI-00788-000', 'BraTS-GLI-01054-000', 'BraTS-GLI-01375-000', 'BraTS-GLI-01374-000', 'BraTS-GLI-00059-001', 
        #         'BraTS-GLI-01484-000', 'BraTS-GLI-01053-000', 'BraTS-GLI-01491-000', 'BraTS-GLI-01083-000', 'BraTS-GLI-00053-001', 'BraTS-GLI-00725-001', 
        #         'BraTS-GLI-00373-000', 'BraTS-GLI-01040-000', 'BraTS-GLI-01532-000', 'BraTS-GLI-01218-000', 'BraTS-GLI-01288-000', 'BraTS-GLI-00369-000', 
        #         'BraTS-GLI-00152-000', 'BraTS-GLI-01360-000', 'BraTS-GLI-00510-001', 'BraTS-GLI-01103-000', 'BraTS-GLI-00780-000', 'BraTS-GLI-00563-000', 
        #         'BraTS-GLI-00548-000', 'BraTS-GLI-01056-000', 'BraTS-GLI-01088-000', 'BraTS-GLI-00532-000', 'BraTS-GLI-00266-000', 'BraTS-GLI-00518-000', 
        #         'BraTS-GLI-01485-000', 'BraTS-GLI-01401-000', 'BraTS-GLI-00685-000', 'BraTS-GLI-00320-000', 'BraTS-GLI-01260-000', 'BraTS-GLI-01311-000', 
        #         'BraTS-GLI-00104-000', 'BraTS-GLI-00339-000', 'BraTS-GLI-01042-000', 'BraTS-GLI-01661-000', 'BraTS-GLI-01257-000', 'BraTS-GLI-01079-000', 
        #         'BraTS-GLI-00479-001', 'BraTS-GLI-01140-000', 'BraTS-GLI-00674-000', 'BraTS-GLI-00359-000', 'BraTS-GLI-00708-000', 'BraTS-GLI-00128-000', 
        #         'BraTS-GLI-00036-000', 'BraTS-GLI-00242-000', 'BraTS-GLI-00022-001', 'BraTS-GLI-00753-000', 'BraTS-GLI-01201-000', 'BraTS-GLI-00750-000', 
        #         'BraTS-GLI-00366-000', 'BraTS-GLI-00247-000', 'BraTS-GLI-00238-000', 'BraTS-GLI-00767-000', 'BraTS-GLI-01457-000', 'BraTS-GLI-00620-000', 
        #         'BraTS-GLI-00099-001', 'BraTS-GLI-00625-000', 'BraTS-GLI-00094-000', 'BraTS-GLI-01515-000', 'BraTS-GLI-01350-000', 'BraTS-GLI-00402-000', 
        #         'BraTS-GLI-01166-000', 'BraTS-GLI-01010-000', 'BraTS-GLI-00547-001', 'BraTS-GLI-01113-000', 'BraTS-GLI-01156-000', 'BraTS-GLI-00651-000', 
        #         'BraTS-GLI-00085-000', 'BraTS-GLI-01660-000', 'BraTS-GLI-01228-000', 'BraTS-GLI-01211-000', 'BraTS-GLI-01436-000', 'BraTS-GLI-01334-000', 
        #         'BraTS-GLI-00201-000', 'BraTS-GLI-00511-001', 'BraTS-GLI-00728-000', 'BraTS-GLI-01511-000', 'BraTS-GLI-00676-000', 'BraTS-GLI-01307-000', 
        #         'BraTS-GLI-01380-000', 'BraTS-GLI-01020-000', 'BraTS-GLI-01420-000', 'BraTS-GLI-01179-000', 'BraTS-GLI-01075-000', 'BraTS-GLI-01289-000', 
        #         'BraTS-GLI-00734-001', 'BraTS-GLI-01394-000', 'BraTS-GLI-01371-000', 'BraTS-GLI-00477-001', 'BraTS-GLI-00558-000', 'BraTS-GLI-00689-000', 
        #         'BraTS-GLI-01662-000', 'BraTS-GLI-00806-000', 'BraTS-GLI-01193-000', 'BraTS-GLI-00735-001', 'BraTS-GLI-01505-000', 'BraTS-GLI-00008-001', 
        #         'BraTS-GLI-00630-001', 'BraTS-GLI-00377-000', 'BraTS-GLI-01430-000', 'BraTS-GLI-01158-000', 'BraTS-GLI-00033-000', 'BraTS-GLI-01092-000', 'BraTS-GLI-00233-000', 'BraTS-GLI-00131-000', 'BraTS-GLI-01067-000', 
        #         'BraTS-GLI-01099-000', 'BraTS-GLI-00750-001', 'BraTS-GLI-01206-000', 'BraTS-GLI-00305-000', 'BraTS-GLI-00017-001', 'BraTS-GLI-01217-000', 'BraTS-GLI-00478-001', 'BraTS-GLI-00436-000', 'BraTS-GLI-00151-000', 
        #         'BraTS-GLI-01186-000', 'BraTS-GLI-00098-000', 'BraTS-GLI-01031-000', 'BraTS-GLI-00209-000', 'BraTS-GLI-00353-000', 'BraTS-GLI-00584-000', 'BraTS-GLI-01073-000', 'BraTS-GLI-00679-000', 'BraTS-GLI-00638-000', 
        #         'BraTS-GLI-01437-000', 'BraTS-GLI-00310-000', 'BraTS-GLI-01285-000', 'BraTS-GLI-01448-000', 'BraTS-GLI-00581-000', 'BraTS-GLI-00058-001', 'BraTS-GLI-01379-000', 'BraTS-GLI-00454-000', 'BraTS-GLI-00688-000',
        #         'BraTS-GLI-00593-000', 'BraTS-GLI-01048-000', 'BraTS-GLI-00686-000', 'BraTS-GLI-00006-000', 'BraTS-GLI-00801-000', 'BraTS-GLI-00072-001', 'BraTS-GLI-01080-000', 'BraTS-GLI-01108-000', 'BraTS-GLI-00149-000',
        #         'BraTS-GLI-00578-000', 'BraTS-GLI-00631-000', 'BraTS-GLI-00839-000', 'BraTS-GLI-00545-001', 'BraTS-GLI-00432-000', 'BraTS-GLI-01221-000', 'BraTS-GLI-00483-000', 'BraTS-GLI-00729-001', 'BraTS-GLI-01467-000', 
        #         'BraTS-GLI-01145-000', 'BraTS-GLI-00517-001', 'BraTS-GLI-01499-000', 'BraTS-GLI-00759-000', 'BraTS-GLI-00089-000', 'BraTS-GLI-00097-001', 'BraTS-GLI-00520-001', 'BraTS-GLI-01255-000', 'BraTS-GLI-01003-000', 
        #         'BraTS-GLI-01399-000', 'BraTS-GLI-00774-001', 'BraTS-GLI-01496-000', 'BraTS-GLI-01246-000', 'BraTS-GLI-00481-000', 'BraTS-GLI-01163-000', 'BraTS-GLI-00772-000', 'BraTS-GLI-00115-000', 'BraTS-GLI-00709-000', 
        #         'BraTS-GLI-01265-000', 'BraTS-GLI-00744-000', 'BraTS-GLI-01381-000', 'BraTS-GLI-01487-000', 'BraTS-GLI-01119-000', 'BraTS-GLI-01098-000', 'BraTS-GLI-01320-000', 'BraTS-GLI-01036-000', 'BraTS-GLI-01370-000', 
        #         'BraTS-GLI-01043-000', 'BraTS-GLI-01055-000', 'BraTS-GLI-01188-000', 'BraTS-GLI-01359-000', 'BraTS-GLI-00731-000', 'BraTS-GLI-01290-000', 'BraTS-GLI-00288-000', 'BraTS-GLI-00684-000', 'BraTS-GLI-00575-000', 
        #         'BraTS-GLI-00014-001', 'BraTS-GLI-00409-000', 'BraTS-GLI-00220-000', 'BraTS-GLI-01090-000', 'BraTS-GLI-01182-000', 'BraTS-GLI-01330-000', 'BraTS-GLI-01454-000', 'BraTS-GLI-01231-000', 'BraTS-GLI-01038-000', 
        #         'BraTS-GLI-00698-000', 'BraTS-GLI-00510-000', 'BraTS-GLI-01234-000', 'BraTS-GLI-01489-000', 'BraTS-GLI-00655-000', 'BraTS-GLI-00132-000', 'BraTS-GLI-01129-000', 'BraTS-GLI-00652-000', 'BraTS-GLI-00376-000', 
        #         'BraTS-GLI-01478-000', 'BraTS-GLI-01247-000', 'BraTS-GLI-00768-000', 'BraTS-GLI-00322-000', 'BraTS-GLI-00395-000', 'BraTS-GLI-00367-000', 'BraTS-GLI-01659-000', 'BraTS-GLI-00054-000', 'BraTS-GLI-00262-000', 
        #         'BraTS-GLI-00101-000', 'BraTS-GLI-00210-000', 'BraTS-GLI-00526-000', 'BraTS-GLI-00824-000', 'BraTS-GLI-01086-000', 'BraTS-GLI-00551-000', 'BraTS-GLI-01267-000', 'BraTS-GLI-01313-000', 'BraTS-GLI-01497-000', 
        #         'BraTS-GLI-00193-000', 'BraTS-GLI-00290-000', 'BraTS-GLI-01353-000', 'BraTS-GLI-00188-000', 'BraTS-GLI-01015-000', 'BraTS-GLI-00556-000', 'BraTS-GLI-00204-000', 'BraTS-GLI-00103-000', 'BraTS-GLI-01244-000', 
        #         'BraTS-GLI-00346-000', 'BraTS-GLI-00623-000', 'BraTS-GLI-01276-000', 'BraTS-GLI-00270-000', 'BraTS-GLI-00031-001', 'BraTS-GLI-01341-000', 'BraTS-GLI-01001-000', 'BraTS-GLI-01046-000', 'BraTS-GLI-01256-000', 
        #         'BraTS-GLI-00212-000', 'BraTS-GLI-00668-000', 'BraTS-GLI-00481-001', 'BraTS-GLI-01286-000', 'BraTS-GLI-00555-001', 'BraTS-GLI-01277-000', 'BraTS-GLI-00203-000', 'BraTS-GLI-00691-000', 'BraTS-GLI-00301-000', 
        #         'BraTS-GLI-01245-000', 'BraTS-GLI-00495-000', 'BraTS-GLI-01435-000', 'BraTS-GLI-00271-000', 'BraTS-GLI-00351-000', 'BraTS-GLI-00693-000', 'BraTS-GLI-01327-000', 'BraTS-GLI-00291-000', 'BraTS-GLI-01530-000', 
        #         'BraTS-GLI-00214-000', 'BraTS-GLI-00552-001', 'BraTS-GLI-01282-000', 'BraTS-GLI-01102-000', 'BraTS-GLI-00597-000', 'BraTS-GLI-00589-000', 'BraTS-GLI-00096-001', 'BraTS-GLI-00028-000', 'BraTS-GLI-01521-000', 
        #         'BraTS-GLI-01344-000', 'BraTS-GLI-00733-000', 'BraTS-GLI-00127-000', 'BraTS-GLI-00018-000', 'BraTS-GLI-00239-000', 'BraTS-GLI-00705-000', 'BraTS-GLI-00636-000', 'BraTS-GLI-00607-001', 'BraTS-GLI-01084-000', 
        #         'BraTS-GLI-00550-001', 'BraTS-GLI-00118-000', 'BraTS-GLI-01050-000', 'BraTS-GLI-00312-000', 'BraTS-GLI-01314-000', 'BraTS-GLI-01026-000', 'BraTS-GLI-00619-001', 'BraTS-GLI-01225-000', 'BraTS-GLI-01451-000', 
        #         'BraTS-GLI-00111-000', 'BraTS-GLI-00070-000', 'BraTS-GLI-01367-000', 'BraTS-GLI-01177-000', 'BraTS-GLI-00646-000', 'BraTS-GLI-01120-000', 'BraTS-GLI-00084-001', 'BraTS-GLI-00469-000', 'BraTS-GLI-00805-000', 
        #         'BraTS-GLI-00263-000', 'BraTS-GLI-01019-000', 'BraTS-GLI-01076-000', 'BraTS-GLI-01132-000', 'BraTS-GLI-00196-000', 'BraTS-GLI-00784-000', 'BraTS-GLI-01142-000', 'BraTS-GLI-01014-000', 'BraTS-GLI-00102-000', 
        #         'BraTS-GLI-01482-000', 'BraTS-GLI-00530-000', 'BraTS-GLI-00612-001', 'BraTS-GLI-00791-000', 'BraTS-GLI-00622-000', 'BraTS-GLI-01349-000', 'BraTS-GLI-01125-000', 'BraTS-GLI-00002-000', 'BraTS-GLI-00406-000', 
        #         'BraTS-GLI-01095-000', 'BraTS-GLI-00661-000', 'BraTS-GLI-00219-000', 'BraTS-GLI-01328-000', 'BraTS-GLI-00568-000', 'BraTS-GLI-01171-000', 'BraTS-GLI-00472-000', 'BraTS-GLI-01058-000', 'BraTS-GLI-01181-000', 
        #         'BraTS-GLI-00540-000', 'BraTS-GLI-01199-000', 'BraTS-GLI-00008-000', 'BraTS-GLI-00241-000', 'BraTS-GLI-00583-000', 'BraTS-GLI-00655-001', 'BraTS-GLI-00016-000', 'BraTS-GLI-00116-000', 'BraTS-GLI-00407-000', 
        #         'BraTS-GLI-00024-001', 'BraTS-GLI-01414-000', 'BraTS-GLI-00081-000', 'BraTS-GLI-01205-000', 'BraTS-GLI-00639-000', 'BraTS-GLI-00727-000', 'BraTS-GLI-00430-000', 'BraTS-GLI-00313-000', 'BraTS-GLI-00187-000', 
        #         'BraTS-GLI-00318-000', 'BraTS-GLI-01271-000', 'BraTS-GLI-00311-000', 'BraTS-GLI-01007-000', 'BraTS-GLI-01292-000', 'BraTS-GLI-00641-000', 'BraTS-GLI-00636-001', 'BraTS-GLI-00675-001', 'BraTS-GLI-00572-000', 
        #         'BraTS-GLI-01537-000', 'BraTS-GLI-01346-000', 'BraTS-GLI-00567-000', 'BraTS-GLI-01296-000', 'BraTS-GLI-00077-000', 'BraTS-GLI-01027-001', 'BraTS-GLI-00387-000', 'BraTS-GLI-00249-000', 'BraTS-GLI-01475-000', 
        #         'BraTS-GLI-01283-000', 'BraTS-GLI-00725-000', 'BraTS-GLI-01011-000', 'BraTS-GLI-00730-000', 'BraTS-GLI-00074-000', 'BraTS-GLI-00528-000', 'BraTS-GLI-00370-000', 'BraTS-GLI-00230-000', 'BraTS-GLI-01453-000', 
        #         'BraTS-GLI-00746-000', 'BraTS-GLI-00155-000', 'BraTS-GLI-01068-000', 'BraTS-GLI-01416-000', 'BraTS-GLI-00491-001', 'BraTS-GLI-01476-000', 'BraTS-GLI-00820-000', 'BraTS-GLI-01386-000', 'BraTS-GLI-01262-000', 
        #         'BraTS-GLI-00056-001', 'BraTS-GLI-00341-000', 'BraTS-GLI-00737-000', 'BraTS-GLI-00412-000', 'BraTS-GLI-01402-000', 'BraTS-GLI-01005-000', 'BraTS-GLI-00072-000', 'BraTS-GLI-01464-000', 'BraTS-GLI-00999-000', 
        #         'BraTS-GLI-00162-000', 'BraTS-GLI-00714-001', 'BraTS-GLI-01044-000', 'BraTS-GLI-01390-000', 'BraTS-GLI-00576-000', 'BraTS-GLI-00787-000', 'BraTS-GLI-01357-000', 'BraTS-GLI-00061-001', 'BraTS-GLI-01516-000', 
        #         'BraTS-GLI-00343-000', 'BraTS-GLI-00177-000', 'BraTS-GLI-00146-000', 'BraTS-GLI-00514-000', 'BraTS-GLI-00830-000', 'BraTS-GLI-00019-000', 'BraTS-GLI-01508-000', 'BraTS-GLI-00389-000', 'BraTS-GLI-01153-000', 
        #         'BraTS-GLI-00446-000', 'BraTS-GLI-00765-000', 'BraTS-GLI-01450-000', 'BraTS-GLI-00840-000', 'BraTS-GLI-00139-000', 'BraTS-GLI-01470-000', 'BraTS-GLI-00056-000', 'BraTS-GLI-00283-000', 'BraTS-GLI-01281-000', 
        #         'BraTS-GLI-01160-001', 'BraTS-GLI-00379-000', 'BraTS-GLI-00338-000', 'BraTS-GLI-01299-000', 'BraTS-GLI-00142-000', 'BraTS-GLI-00606-000', 'BraTS-GLI-00811-000', 'BraTS-GLI-01333-000', 'BraTS-GLI-01500-000', 
        #         'BraTS-GLI-00150-000', 'BraTS-GLI-00795-000', 'BraTS-GLI-00014-000', 'BraTS-GLI-00611-000', 'BraTS-GLI-01406-000', 'BraTS-GLI-01408-000', 'BraTS-GLI-00500-000', 'BraTS-GLI-00349-000', 'BraTS-GLI-01178-000', 
        #         'BraTS-GLI-01443-000', 'BraTS-GLI-00751-001', 'BraTS-GLI-00451-000', 'BraTS-GLI-00667-000', 'BraTS-GLI-00299-000', 'BraTS-GLI-01165-000', 'BraTS-GLI-01041-000', 'BraTS-GLI-00267-000', 'BraTS-GLI-00658-000', 
        #         'BraTS-GLI-00400-000', 'BraTS-GLI-00350-000', 'BraTS-GLI-01105-000', 'BraTS-GLI-01258-000', 'BraTS-GLI-01013-000', 'BraTS-GLI-00060-000', 'BraTS-GLI-01052-000', 'BraTS-GLI-00676-001', 'BraTS-GLI-01473-000', 
        #         'BraTS-GLI-00797-000', 'BraTS-GLI-00405-000', 'BraTS-GLI-00285-000', 'BraTS-GLI-00466-000', 'BraTS-GLI-00735-000', 'BraTS-GLI-01025-000', 'BraTS-GLI-00803-000', 'BraTS-GLI-00109-000', 'BraTS-GLI-01424-000', 
        #         'BraTS-GLI-00306-000', 'BraTS-GLI-01417-000', 'BraTS-GLI-00418-000', 'BraTS-GLI-01146-000', 'BraTS-GLI-01136-000', 'BraTS-GLI-00216-000', 'BraTS-GLI-01214-000', 'BraTS-GLI-01137-000', 'BraTS-GLI-00588-000', 
        #         'BraTS-GLI-00206-000', 'BraTS-GLI-00289-000', 'BraTS-GLI-01362-000', 'BraTS-GLI-00708-001', 'BraTS-GLI-00211-000', 'BraTS-GLI-00550-000', 'BraTS-GLI-01412-000', 'BraTS-GLI-00630-000', 'BraTS-GLI-01393-000', 
        #         'BraTS-GLI-01347-000', 'BraTS-GLI-00178-000', 'BraTS-GLI-00121-000', 'BraTS-GLI-00356-000', 'BraTS-GLI-01239-000', 'BraTS-GLI-00228-000', 'BraTS-GLI-01514-000', 'BraTS-GLI-00816-000', 'BraTS-GLI-00707-000', 
        #         'BraTS-GLI-01089-000', 'BraTS-GLI-00253-000', 'BraTS-GLI-01438-000', 'BraTS-GLI-00421-000', 'BraTS-GLI-01039-000', 'BraTS-GLI-01194-000', 'BraTS-GLI-01518-000', 'BraTS-GLI-01452-000', 'BraTS-GLI-00061-000', 
        #         'BraTS-GLI-01415-000', 'BraTS-GLI-00774-000', 'BraTS-GLI-01425-000', 'BraTS-GLI-01398-000', 'BraTS-GLI-00477-000', 'BraTS-GLI-00390-000', 'BraTS-GLI-00740-000', 'BraTS-GLI-00703-001', 'BraTS-GLI-00538-000', 
        #         'BraTS-GLI-01298-000', 'BraTS-GLI-00808-000', 'BraTS-GLI-01238-000', 'BraTS-GLI-01480-000', 'BraTS-GLI-01536-000', 'BraTS-GLI-01152-000', 'BraTS-GLI-00021-000', 'BraTS-GLI-00807-000', 'BraTS-GLI-00051-000', 
        #         'BraTS-GLI-00540-001', 'BraTS-GLI-01240-000', 'BraTS-GLI-01361-000', 'BraTS-GLI-01151-000', 'BraTS-GLI-00300-000', 'BraTS-GLI-00309-000', 'BraTS-GLI-00628-000', 'BraTS-GLI-01316-000', 'BraTS-GLI-00284-000', 
        #         'BraTS-GLI-01502-000', 'BraTS-GLI-00166-000', 'BraTS-GLI-01324-000', 'BraTS-GLI-00185-000', 'BraTS-GLI-01202-000', 'BraTS-GLI-00657-000', 'BraTS-GLI-00110-000', 'BraTS-GLI-01664-000', 'BraTS-GLI-00227-000', 
        #         'BraTS-GLI-00431-000', 'BraTS-GLI-00683-001', 'BraTS-GLI-01272-000', 'BraTS-GLI-01431-000', 'BraTS-GLI-00594-000', 'BraTS-GLI-01291-000', 'BraTS-GLI-01226-000', 'BraTS-GLI-00261-000', 'BraTS-GLI-01127-000', 
        #         'BraTS-GLI-01059-000', 'BraTS-GLI-00329-000', 'BraTS-GLI-00456-000', 'BraTS-GLI-01498-000', 'BraTS-GLI-00122-000', 'BraTS-GLI-00571-000', 'BraTS-GLI-01509-000', 'BraTS-GLI-00158-000', 'BraTS-GLI-00296-000', 
        #         'BraTS-GLI-01369-000', 'BraTS-GLI-01139-000', 'BraTS-GLI-00778-000', 'BraTS-GLI-00009-000', 'BraTS-GLI-00136-000', 'BraTS-GLI-00254-000', 'BraTS-GLI-01315-000', 'BraTS-GLI-01117-000', 'BraTS-GLI-00159-000', 
        #         'BraTS-GLI-00729-000', 'BraTS-GLI-00483-001', 'BraTS-GLI-01446-000', 'BraTS-GLI-00012-000', 'BraTS-GLI-01237-000', 'BraTS-GLI-00464-001', 'BraTS-GLI-00457-000', 'BraTS-GLI-00732-000', 'BraTS-GLI-01131-000', 
        #         'BraTS-GLI-01009-000', 'BraTS-GLI-01197-000', 'BraTS-GLI-00469-001', 'BraTS-GLI-01396-000', 'BraTS-GLI-00321-000', 'BraTS-GLI-00429-000', 'BraTS-GLI-00616-000', 'BraTS-GLI-01321-000', 'BraTS-GLI-01384-000', 
        #         'BraTS-GLI-01133-000', 'BraTS-GLI-00561-000', 'BraTS-GLI-00031-000', 'BraTS-GLI-01072-000', 'BraTS-GLI-01029-000', 'BraTS-GLI-01189-000', 'BraTS-GLI-01343-000', 'BraTS-GLI-00537-000', 'BraTS-GLI-01525-000', 
        #         'BraTS-GLI-01064-000', 'BraTS-GLI-00544-000', 'BraTS-GLI-00692-001', 'BraTS-GLI-01037-000', 'BraTS-GLI-01112-000', 'BraTS-GLI-00123-000', 'BraTS-GLI-00401-000', 'BraTS-GLI-01081-000', 'BraTS-GLI-00542-000', 
        #         'BraTS-GLI-00649-000', 'BraTS-GLI-01198-000', 'BraTS-GLI-01223-000', 'BraTS-GLI-00292-000', 'BraTS-GLI-01114-000', 'BraTS-GLI-00703-000', 'BraTS-GLI-00694-001', 'BraTS-GLI-01428-000', 'BraTS-GLI-01378-000', 
        #         'BraTS-GLI-00325-000', 'BraTS-GLI-01442-000', 'BraTS-GLI-01062-000', 'BraTS-GLI-00048-001', 'BraTS-GLI-00087-000', 'BraTS-GLI-00478-000', 'BraTS-GLI-00574-000', 'BraTS-GLI-01303-000', 'BraTS-GLI-01049-000', 
        #         'BraTS-GLI-00559-000', 'BraTS-GLI-00511-000', 'BraTS-GLI-00043-000', 'BraTS-GLI-00234-000', 'BraTS-GLI-01150-000', 'BraTS-GLI-00133-000', 'BraTS-GLI-01208-000', 'BraTS-GLI-00505-000', 'BraTS-GLI-01008-000', 
        #         'BraTS-GLI-00388-000', 'BraTS-GLI-00756-000', 'BraTS-GLI-01413-000', 'BraTS-GLI-01185-000', 'BraTS-GLI-01519-000', 'BraTS-GLI-00747-000', 'BraTS-GLI-00587-000', 'BraTS-GLI-01085-000', 'BraTS-GLI-00062-000', 
        #         'BraTS-GLI-00120-000', 'BraTS-GLI-00611-001', 'BraTS-GLI-01012-000', 'BraTS-GLI-00240-000', 'BraTS-GLI-00459-000', 'BraTS-GLI-01490-000', 'BraTS-GLI-00680-001', 'BraTS-GLI-01104-000', 'BraTS-GLI-01385-000', 
        #         'BraTS-GLI-00767-001', 'BraTS-GLI-00147-000', 'BraTS-GLI-00751-000', 'BraTS-GLI-01312-000', 'BraTS-GLI-01441-000', 'BraTS-GLI-01027-000', 'BraTS-GLI-01355-000', 'BraTS-GLI-01455-000', 'BraTS-GLI-00108-000', 
        #         'BraTS-GLI-00052-000', 'BraTS-GLI-01109-000', 'BraTS-GLI-00504-000', 'BraTS-GLI-00586-000', 'BraTS-GLI-01305-000', 'BraTS-GLI-01434-000', 'BraTS-GLI-00529-000', 'BraTS-GLI-00470-000', 'BraTS-GLI-00176-000', 
        #         'BraTS-GLI-00823-000', 'BraTS-GLI-01287-000', 'BraTS-GLI-01403-000', 'BraTS-GLI-00612-000', 'BraTS-GLI-00157-000', 'BraTS-GLI-01657-000', 'BraTS-GLI-01520-000', 'BraTS-GLI-01077-000', 'BraTS-GLI-00059-000', 
        #         'BraTS-GLI-00444-000', 'BraTS-GLI-00375-000', 'BraTS-GLI-00303-000', 'BraTS-GLI-01337-000', 'BraTS-GLI-01183-000', 'BraTS-GLI-01216-000', 'BraTS-GLI-00064-000', 'BraTS-GLI-00501-000', 'BraTS-GLI-01351-000', 
        #         'BraTS-GLI-00525-000', 'BraTS-GLI-00399-000', 'BraTS-GLI-00336-000', 'BraTS-GLI-00613-000', 'BraTS-GLI-01061-000', 'BraTS-GLI-00442-000', 'BraTS-GLI-00324-000', 'BraTS-GLI-00697-000', 'BraTS-GLI-00626-000', 
        #         'BraTS-GLI-01173-000', 'BraTS-GLI-01445-000', 'BraTS-GLI-01332-000', 'BraTS-GLI-00800-000', 'BraTS-GLI-00020-001', 'BraTS-GLI-01439-000', 'BraTS-GLI-00723-000', 'BraTS-GLI-01249-000', 'BraTS-GLI-00506-000', 
        #         'BraTS-GLI-00605-000', 'BraTS-GLI-01317-000', 'BraTS-GLI-00731-001', 'BraTS-GLI-00659-000', 'BraTS-GLI-01325-000', 'BraTS-GLI-00259-000', 'BraTS-GLI-01200-000', 'BraTS-GLI-00087-001', 'BraTS-GLI-00732-001', 
        #         'BraTS-GLI-01336-000', 'BraTS-GLI-01318-000', 'BraTS-GLI-01517-000', 'BraTS-GLI-00704-000', 'BraTS-GLI-00032-001', 'BraTS-GLI-00621-000', 'BraTS-GLI-00654-001', 'BraTS-GLI-01501-000', 'BraTS-GLI-00624-000', 
        #         'BraTS-GLI-00493-000', 'BraTS-GLI-00579-000', 'BraTS-GLI-01248-000', 'BraTS-GLI-00170-000', 'BraTS-GLI-01461-000', 'BraTS-GLI-00440-000', 'BraTS-GLI-01148-000', 'BraTS-GLI-00017-000', 'BraTS-GLI-00392-000', 
        #         'BraTS-GLI-01377-000', 'BraTS-GLI-00144-000', 'BraTS-GLI-00286-000', 'BraTS-GLI-01065-000', 'BraTS-GLI-00044-000', 'BraTS-GLI-00640-000', 'BraTS-GLI-01184-000', 'BraTS-GLI-01161-000', 'BraTS-GLI-00590-000', 
        #         'BraTS-GLI-01023-000', 'BraTS-GLI-01224-000', 'BraTS-GLI-00386-000', 'BraTS-GLI-01486-000', 'BraTS-GLI-00499-000', 'BraTS-GLI-01266-000', 'BraTS-GLI-01488-000', 'BraTS-GLI-00328-000', 'BraTS-GLI-01101-000', 
        #         'BraTS-GLI-00649-001', 'BraTS-GLI-01219-000', 'BraTS-GLI-00423-000', 'BraTS-GLI-00417-000', 'BraTS-GLI-01063-000', 'BraTS-GLI-01022-000', 'BraTS-GLI-00452-000', 'BraTS-GLI-01232-000', 'BraTS-GLI-00280-000', 
        #         'BraTS-GLI-00739-000', 'BraTS-GLI-01440-000', 'BraTS-GLI-01342-000', 'BraTS-GLI-01523-000', 'BraTS-GLI-00773-000', 'BraTS-GLI-01091-000', 'BraTS-GLI-00619-000', 'BraTS-GLI-00425-000', 'BraTS-GLI-01143-000', 
        #         'BraTS-GLI-00615-000', 'BraTS-GLI-01275-000', 'BraTS-GLI-01506-000', 'BraTS-GLI-00837-000', 'BraTS-GLI-00775-000', 'BraTS-GLI-00523-000', 'BraTS-GLI-01534-000', 'BraTS-GLI-01427-000', 'BraTS-GLI-01504-000', 
        #         'BraTS-GLI-00730-001', 'BraTS-GLI-01111-000', 'BraTS-GLI-01230-000', 'BraTS-GLI-00677-000', 'BraTS-GLI-00480-001', 'BraTS-GLI-01471-000', 'BraTS-GLI-01243-000', 'BraTS-GLI-01159-000', 'BraTS-GLI-00112-000', 
        #         'BraTS-GLI-00165-000', 'BraTS-GLI-00757-000', 'BraTS-GLI-00273-000', 'BraTS-GLI-01253-000', 'BraTS-GLI-00777-001', 'BraTS-GLI-01236-000', 'BraTS-GLI-01278-000', 'BraTS-GLI-00516-000', 'BraTS-GLI-01176-000', 
        #         'BraTS-GLI-01302-000', 'BraTS-GLI-01395-000', 'BraTS-GLI-00831-000', 'BraTS-GLI-01069-000', 'BraTS-GLI-00569-000', 'BraTS-GLI-00650-000', 'BraTS-GLI-01510-000', 'BraTS-GLI-01250-000', 'BraTS-GLI-00068-000', 
        #         'BraTS-GLI-01207-000', 'BraTS-GLI-00479-000', 'BraTS-GLI-00237-000', 'BraTS-GLI-00549-001', 'BraTS-GLI-01168-000', 'BraTS-GLI-01458-000', 'BraTS-GLI-00718-000', 'BraTS-GLI-01322-000', 'BraTS-GLI-00281-000', 
        #         'BraTS-GLI-00488-000', 'BraTS-GLI-00095-000', 'BraTS-GLI-00544-001', 'BraTS-GLI-00274-000', 'BraTS-GLI-00836-000', 'BraTS-GLI-01373-000', 'BraTS-GLI-00716-000', 'BraTS-GLI-01106-000', 'BraTS-GLI-00610-001', 
        #         'BraTS-GLI-00565-000', 'BraTS-GLI-01213-000', 'BraTS-GLI-00642-000', 'BraTS-GLI-00685-001', 'BraTS-GLI-00512-000', 'BraTS-GLI-01304-000', 'BraTS-GLI-00009-001', 'BraTS-GLI-00502-001', 'BraTS-GLI-00063-000', 
        #         'BraTS-GLI-00282-000', 'BraTS-GLI-01251-000', 'BraTS-GLI-01527-000', 'BraTS-GLI-00095-001', 'BraTS-GLI-01126-000', 'BraTS-GLI-01469-000', 'BraTS-GLI-00097-000', 'BraTS-GLI-00775-001', 'BraTS-GLI-01124-000', 
        #         'BraTS-GLI-01368-000', 'BraTS-GLI-00692-000', 'BraTS-GLI-00760-000', 'BraTS-GLI-01323-000', 'BraTS-GLI-00184-000', 'BraTS-GLI-01002-000', 'BraTS-GLI-01051-000', 'BraTS-GLI-00445-000', 'BraTS-GLI-01196-000', 
        #         'BraTS-GLI-00036-001', 'BraTS-GLI-00547-000', 'BraTS-GLI-00618-000', 'BraTS-GLI-01016-000', 'BraTS-GLI-01280-000', 'BraTS-GLI-01348-000', 'BraTS-GLI-01433-000', 'BraTS-GLI-00455-000', 'BraTS-GLI-01130-000', 
        #         'BraTS-GLI-01116-000', 'BraTS-GLI-01665-000', 'BraTS-GLI-01227-000', 'BraTS-GLI-00596-000', 'BraTS-GLI-00604-000', 'BraTS-GLI-00689-001', 'BraTS-GLI-00106-000', 'BraTS-GLI-00117-000', 'BraTS-GLI-00533-000', 
        #         'BraTS-GLI-01463-000', 'BraTS-GLI-00433-000', 'BraTS-GLI-01154-000', 'BraTS-GLI-00519-000', 'BraTS-GLI-01297-000', 'BraTS-GLI-00078-000', 'BraTS-GLI-01045-000', 'BraTS-GLI-00646-001', 'BraTS-GLI-01175-000', 
        #         'BraTS-GLI-01465-000', 'BraTS-GLI-00404-000', 'BraTS-GLI-00053-000', 'BraTS-GLI-01423-000', 'BraTS-GLI-01047-000', 'BraTS-GLI-00601-000', 'BraTS-GLI-01319-000', 'BraTS-GLI-01529-000', 'BraTS-GLI-00143-000', 
        #         'BraTS-GLI-00680-000', 'BraTS-GLI-00513-000', 'BraTS-GLI-01400-000', 'BraTS-GLI-00378-000', 'BraTS-GLI-00715-001', 'BraTS-GLI-00171-000', 'BraTS-GLI-00194-000', 'BraTS-GLI-01363-000', 'BraTS-GLI-00045-001', 
        #         'BraTS-GLI-00035-000', 'BraTS-GLI-01356-000', 'BraTS-GLI-01097-000', 'BraTS-GLI-01389-000', 'BraTS-GLI-01340-000', 'BraTS-GLI-01358-000', 'BraTS-GLI-01472-000', 'BraTS-GLI-01107-000', 'BraTS-GLI-00448-000', 
        #         'BraTS-GLI-00443-000', 'BraTS-GLI-00348-000', 'BraTS-GLI-00172-000', 'BraTS-GLI-01466-000', 'BraTS-GLI-00011-000', 'BraTS-GLI-00380-000', 'BraTS-GLI-00724-000', 'BraTS-GLI-00495-001', 'BraTS-GLI-00260-000', 
        #         'BraTS-GLI-00217-000', 'BraTS-GLI-00663-000', 'BraTS-GLI-01269-000', 'BraTS-GLI-00494-000', 'BraTS-GLI-00552-000', 'BraTS-GLI-01610-000', 'BraTS-GLI-00577-000', 'BraTS-GLI-01222-000', 'BraTS-GLI-00781-000', 
        #         'BraTS-GLI-01028-000', 'BraTS-GLI-00613-001', 'BraTS-GLI-00799-000', 'BraTS-GLI-01309-000', 'BraTS-GLI-01018-000', 'BraTS-GLI-01066-000', 'BraTS-GLI-01164-000', 'BraTS-GLI-01524-000', 'BraTS-GLI-01167-000', 
        #         'BraTS-GLI-01229-000', 'BraTS-GLI-00694-000', 'BraTS-GLI-00485-000', 'BraTS-GLI-01335-000', 'BraTS-GLI-00687-000', 'BraTS-GLI-00331-000', 'BraTS-GLI-00838-000', 'BraTS-GLI-00258-000', 'BraTS-GLI-00045-000', 
        #         'BraTS-GLI-00046-000', 'BraTS-GLI-00020-000', 'BraTS-GLI-01409-000', 'BraTS-GLI-01535-000', 'BraTS-GLI-00183-000', 'BraTS-GLI-01190-000', 'BraTS-GLI-00124-000', 'BraTS-GLI-00464-000', 'BraTS-GLI-00782-000', 
        #         'BraTS-GLI-01503-000', 'BraTS-GLI-01135-000', 'BraTS-GLI-01513-000', 'BraTS-GLI-00796-000', 'BraTS-GLI-00148-000', 'BraTS-GLI-00327-000', 'BraTS-GLI-01180-000', 'BraTS-GLI-01387-000', 'BraTS-GLI-01110-000', 
        #         'BraTS-GLI-01118-000', 'BraTS-GLI-00570-000', 'BraTS-GLI-01233-000', 'BraTS-GLI-00352-000', 'BraTS-GLI-00021-001', 'BraTS-GLI-00090-001', 'BraTS-GLI-00764-000', 'BraTS-GLI-00514-001', 'BraTS-GLI-01242-000', 
        #         'BraTS-GLI-01172-000', 'BraTS-GLI-00598-000', 'BraTS-GLI-01326-000', 'BraTS-GLI-01494-000', 'BraTS-GLI-01004-000', 'BraTS-GLI-01533-000', 'BraTS-GLI-00397-000'])
        #     splits[self.fold]['val'] = np.array(
        #         ['BraTS-GLI-01192-000', 'BraTS-GLI-00246-000', 'BraTS-GLI-01447-000', 'BraTS-GLI-00736-001', 'BraTS-GLI-00130-000', 'BraTS-GLI-00236-000', 'BraTS-GLI-00105-000', 'BraTS-GLI-00742-000', 'BraTS-GLI-00654-000', 
        #         'BraTS-GLI-00828-000', 'BraTS-GLI-01141-000', 'BraTS-GLI-01301-000', 'BraTS-GLI-01507-000', 'BraTS-GLI-01071-000', 'BraTS-GLI-01391-000', 'BraTS-GLI-00016-001', 'BraTS-GLI-00269-000', 'BraTS-GLI-01352-000', 
        #         'BraTS-GLI-00414-000', 'BraTS-GLI-00539-000', 'BraTS-GLI-00834-000', 'BraTS-GLI-01270-000', 'BraTS-GLI-00518-001', 'BraTS-GLI-00030-000', 'BraTS-GLI-00334-000', 'BraTS-GLI-00199-000', 'BraTS-GLI-00496-000', 
        #         'BraTS-GLI-01149-000', 'BraTS-GLI-00107-000', 'BraTS-GLI-01397-000', 'BraTS-GLI-00758-000', 'BraTS-GLI-00793-000', 'BraTS-GLI-01310-000', 'BraTS-GLI-01462-000', 'BraTS-GLI-00000-000', 'BraTS-GLI-01295-000', 
        #         'BraTS-GLI-01212-000', 'BraTS-GLI-00413-000', 'BraTS-GLI-01663-000', 'BraTS-GLI-00140-000', 'BraTS-GLI-01210-000', 'BraTS-GLI-00441-000', 'BraTS-GLI-00100-000', 'BraTS-GLI-01155-000', 'BraTS-GLI-01122-000', 
        #         'BraTS-GLI-00066-000', 'BraTS-GLI-01354-000', 'BraTS-GLI-01074-000', 'BraTS-GLI-01495-000', 'BraTS-GLI-00559-001', 'BraTS-GLI-00602-000', 'BraTS-GLI-00419-000', 'BraTS-GLI-00814-000', 'BraTS-GLI-00502-000', 
        #         'BraTS-GLI-00005-000', 'BraTS-GLI-00192-000', 'BraTS-GLI-01147-000', 'BraTS-GLI-01093-000', 'BraTS-GLI-01528-000', 'BraTS-GLI-00340-000', 'BraTS-GLI-01034-000', 'BraTS-GLI-01479-000', 'BraTS-GLI-01021-000', 
        #         'BraTS-GLI-01284-000', 'BraTS-GLI-00602-001', 'BraTS-GLI-00347-000', 'BraTS-GLI-00690-000', 'BraTS-GLI-00099-000', 'BraTS-GLI-00819-000', 'BraTS-GLI-00186-000', 'BraTS-GLI-01235-000', 'BraTS-GLI-00024-000', 
        #         'BraTS-GLI-00403-000', 'BraTS-GLI-01383-000', 'BraTS-GLI-01094-000', 'BraTS-GLI-01449-000', 'BraTS-GLI-00756-001', 'BraTS-GLI-01339-000', 'BraTS-GLI-01138-000', 'BraTS-GLI-01263-000', 'BraTS-GLI-00025-000', 
        #         'BraTS-GLI-00468-000', 'BraTS-GLI-01203-000', 'BraTS-GLI-00298-000', 'BraTS-GLI-01531-000', 'BraTS-GLI-01382-000', 'BraTS-GLI-01000-000', 'BraTS-GLI-00426-000', 'BraTS-GLI-00416-000', 'BraTS-GLI-00126-000', 
        #         'BraTS-GLI-01481-000', 'BraTS-GLI-00449-000', 'BraTS-GLI-01261-000', 'BraTS-GLI-01033-000', 'BraTS-GLI-01407-000', 'BraTS-GLI-00517-000', 'BraTS-GLI-01422-000', 'BraTS-GLI-01432-000', 'BraTS-GLI-00088-001', 
        #         'BraTS-GLI-01254-000', 'BraTS-GLI-01329-000', 'BraTS-GLI-01411-000', 'BraTS-GLI-00480-000', 'BraTS-GLI-01191-000', 'BraTS-GLI-01017-000', 'BraTS-GLI-00733-001', 'BraTS-GLI-00160-000', 'BraTS-GLI-00557-000', 
        #         'BraTS-GLI-00156-000', 'BraTS-GLI-01468-000', 'BraTS-GLI-00251-000', 'BraTS-GLI-01169-000', 'BraTS-GLI-00049-000', 'BraTS-GLI-00134-000', 'BraTS-GLI-00543-000', 'BraTS-GLI-01279-000', 'BraTS-GLI-00314-000', 
        #         'BraTS-GLI-00491-000', 'BraTS-GLI-00507-000', 'BraTS-GLI-00048-000', 'BraTS-GLI-00084-000', 'BraTS-GLI-00294-000', 'BraTS-GLI-01057-000', 'BraTS-GLI-01477-000', 'BraTS-GLI-01459-000', 'BraTS-GLI-00304-000', 
        #         'BraTS-GLI-00498-000', 'BraTS-GLI-00316-000', 'BraTS-GLI-00058-000', 'BraTS-GLI-01526-000', 'BraTS-GLI-01392-000', 'BraTS-GLI-00098-001', 'BraTS-GLI-00555-000', 'BraTS-GLI-00382-000', 'BraTS-GLI-01522-000', 
        #         'BraTS-GLI-01512-000', 'BraTS-GLI-01170-000', 'BraTS-GLI-01483-000', 'BraTS-GLI-00453-000', 'BraTS-GLI-01060-000', 'BraTS-GLI-00556-001', 'BraTS-GLI-00297-000', 'BraTS-GLI-00675-000', 'BraTS-GLI-00032-000', 
        #         'BraTS-GLI-00608-000', 'BraTS-GLI-00499-001', 'BraTS-GLI-00818-000', 'BraTS-GLI-01294-000', 'BraTS-GLI-00293-000', 'BraTS-GLI-00789-000', 'BraTS-GLI-00195-000', 'BraTS-GLI-01030-000', 'BraTS-GLI-01306-000', 
        #         'BraTS-GLI-01220-000', 'BraTS-GLI-00682-000', 'BraTS-GLI-01157-000', 'BraTS-GLI-00549-000', 'BraTS-GLI-00591-000', 'BraTS-GLI-01366-000', 'BraTS-GLI-00682-001', 'BraTS-GLI-01372-000', 'BraTS-GLI-01100-000', 
        #         'BraTS-GLI-01405-000', 'BraTS-GLI-01493-000', 'BraTS-GLI-00088-000', 'BraTS-GLI-00391-000', 'BraTS-GLI-00332-000', 'BraTS-GLI-00218-000', 'BraTS-GLI-00154-000', 'BraTS-GLI-01273-000', 'BraTS-GLI-00645-000', 
        #         'BraTS-GLI-00071-000', 'BraTS-GLI-00494-001', 'BraTS-GLI-00498-001', 'BraTS-GLI-00485-001', 'BraTS-GLI-00714-000', 'BraTS-GLI-00607-000', 'BraTS-GLI-00772-001', 'BraTS-GLI-01666-000', 'BraTS-GLI-00222-000', 
        #         'BraTS-GLI-01209-000', 'BraTS-GLI-00706-000', 'BraTS-GLI-01418-000', 'BraTS-GLI-00524-000', 'BraTS-GLI-01404-000', 'BraTS-GLI-00809-000', 'BraTS-GLI-01419-000', 'BraTS-GLI-00317-000', 'BraTS-GLI-00715-000', 
        #         'BraTS-GLI-01426-000', 'BraTS-GLI-00734-000', 'BraTS-GLI-00275-000', 'BraTS-GLI-01300-000', 'BraTS-GLI-01338-000', 'BraTS-GLI-00777-000', 'BraTS-GLI-00191-000', 'BraTS-GLI-00221-000', 'BraTS-GLI-01215-000',
        #         'BraTS-GLI-01410-000', 'BraTS-GLI-00371-000', 'BraTS-GLI-00599-000', 'BraTS-GLI-00558-001', 'BraTS-GLI-01460-000', 'BraTS-GLI-00500-001', 'BraTS-GLI-00138-000', 'BraTS-GLI-00792-000', 'BraTS-GLI-00610-000', 
        #         'BraTS-GLI-00810-000', 'BraTS-GLI-01308-000', 'BraTS-GLI-00003-000', 'BraTS-GLI-01115-000', 'BraTS-GLI-00231-000', 'BraTS-GLI-00026-000', 'BraTS-GLI-01421-000', 'BraTS-GLI-00137-000', 'BraTS-GLI-00645-001', 
        #         'BraTS-GLI-01365-000', 'BraTS-GLI-01204-000', 'BraTS-GLI-01658-000', 'BraTS-GLI-01429-000', 'BraTS-GLI-00525-001', 'BraTS-GLI-00768-001', 'BraTS-GLI-01187-000', 'BraTS-GLI-01024-000', 'BraTS-GLI-00250-000', 
        #         'BraTS-GLI-01444-000', 'BraTS-GLI-01264-000', 'BraTS-GLI-00804-000', 'BraTS-GLI-00167-000', 'BraTS-GLI-00243-000', 'BraTS-GLI-00753-001', 'BraTS-GLI-00383-000', 'BraTS-GLI-01388-000', 'BraTS-GLI-00620-001', 
        #         'BraTS-GLI-00235-000', 'BraTS-GLI-00364-000', 'BraTS-GLI-00691-001', 'BraTS-GLI-00360-000', 'BraTS-GLI-01331-000', 'BraTS-GLI-00554-000', 'BraTS-GLI-01492-000', 'BraTS-GLI-01032-000', 'BraTS-GLI-01274-000', 
        #         'BraTS-GLI-00736-000', 'BraTS-GLI-01128-000', 'BraTS-GLI-01293-000', 'BraTS-GLI-01345-000', 'BraTS-GLI-00641-001', 'BraTS-GLI-00090-000', 'BraTS-GLI-01160-000', 'BraTS-GLI-01087-000'])

        #     split_number = round(len(splits[self.fold]['train']) * (self.split / 100))
        #     print('split percentage:', self.split / 100)
        #     save_pickle(splits, splits_file)
        #     # tr_keys = splits[self.fold]['train']
        #     tr_keys = splits[self.fold]['train'][:split_number]
        #     print('number of training cases:', len(tr_keys))
        #     val_keys = splits[self.fold]['val']
        #     self.print_to_log_file("This split has %d training and %d validation cases."
        #                            % (len(tr_keys), len(val_keys)))

        # tr_keys.sort()
        # val_keys.sort()
        # self.dataset_tr = OrderedDict()
        # for i in tr_keys:
        #     self.dataset_tr[i] = self.dataset[i]
        # self.dataset_val = OrderedDict()
        # for i in val_keys:
        #     self.dataset_val[i] = self.dataset[i]

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
