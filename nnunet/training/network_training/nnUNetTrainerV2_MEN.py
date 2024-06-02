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
from nnunet.network_architecture.generic_UNet import Generic_UNet
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


class nnUNetTrainerV2_MEN(nnUNetTrainer):
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
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
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
                ['BraTS-MEN-00646-000', 'BraTS-MEN-00906-000', 'BraTS-MEN-00238-000', 'BraTS-MEN-00788-000', 
                'BraTS-MEN-00367-000', 'BraTS-MEN-00584-000', 'BraTS-MEN-00955-000', 'BraTS-MEN-00735-000', 
                'BraTS-MEN-00191-000', 'BraTS-MEN-00440-000', 'BraTS-MEN-01401-000', 'BraTS-MEN-00589-000', 
                'BraTS-MEN-00358-000', 'BraTS-MEN-01388-000', 'BraTS-MEN-00540-000', 'BraTS-MEN-00954-000', 
                'BraTS-MEN-01309-000', 'BraTS-MEN-01031-000', 'BraTS-MEN-00465-000', 'BraTS-MEN-00167-000', 
                'BraTS-MEN-00737-000', 'BraTS-MEN-01075-000', 'BraTS-MEN-00831-000', 'BraTS-MEN-00018-000', 
                'BraTS-MEN-01179-000', 'BraTS-MEN-00378-000', 'BraTS-MEN-00547-000', 'BraTS-MEN-01016-000', 
                'BraTS-MEN-00060-000', 'BraTS-MEN-00077-000', 'BraTS-MEN-00601-000', 'BraTS-MEN-00807-000', 
                'BraTS-MEN-00754-000', 'BraTS-MEN-01246-002', 'BraTS-MEN-01246-003', 'BraTS-MEN-00857-000', 
                'BraTS-MEN-00209-000', 'BraTS-MEN-01127-000', 'BraTS-MEN-00530-000', 'BraTS-MEN-01113-000', 
                'BraTS-MEN-01363-000', 'BraTS-MEN-01423-000', 'BraTS-MEN-00196-000', 'BraTS-MEN-00357-000', 
                'BraTS-MEN-00929-000', 'BraTS-MEN-01279-010', 'BraTS-MEN-00886-000', 'BraTS-MEN-01305-000', 
                'BraTS-MEN-00851-000', 'BraTS-MEN-01302-000', 'BraTS-MEN-01408-000', 'BraTS-MEN-01118-000', 
                'BraTS-MEN-00412-000', 'BraTS-MEN-00722-000', 'BraTS-MEN-00781-008', 'BraTS-MEN-00748-000', 
                'BraTS-MEN-00561-000', 'BraTS-MEN-00087-000', 'BraTS-MEN-00229-000', 'BraTS-MEN-01041-000', 
                'BraTS-MEN-01116-000', 'BraTS-MEN-00843-000', 'BraTS-MEN-00684-000', 'BraTS-MEN-00640-000', 
                'BraTS-MEN-00223-000', 'BraTS-MEN-00342-000', 'BraTS-MEN-00792-000', 'BraTS-MEN-01042-000', 
                'BraTS-MEN-00402-000', 'BraTS-MEN-01208-000', 'BraTS-MEN-00366-000', 'BraTS-MEN-00074-006', 
                'BraTS-MEN-01184-000', 'BraTS-MEN-01008-011', 'BraTS-MEN-00153-000', 'BraTS-MEN-00416-000', 
                'BraTS-MEN-00729-000', 'BraTS-MEN-00689-000', 'BraTS-MEN-01015-000', 'BraTS-MEN-00401-000', 
                'BraTS-MEN-00061-000', 'BraTS-MEN-01088-000', 'BraTS-MEN-00745-000', 'BraTS-MEN-00119-000', 
                'BraTS-MEN-00982-000', 'BraTS-MEN-01191-000', 'BraTS-MEN-00588-000', 'BraTS-MEN-00713-000', 
                'BraTS-MEN-01366-000', 'BraTS-MEN-00387-000', 'BraTS-MEN-00766-000', 'BraTS-MEN-01130-000', 
                'BraTS-MEN-00728-000', 'BraTS-MEN-01087-000', 'BraTS-MEN-01385-000', 'BraTS-MEN-01241-000', 
                'BraTS-MEN-01054-000', 'BraTS-MEN-00960-000', 'BraTS-MEN-00492-000', 'BraTS-MEN-00619-000', 
                'BraTS-MEN-00192-000', 'BraTS-MEN-00419-000', 'BraTS-MEN-00446-000', 'BraTS-MEN-00102-000', 
                'BraTS-MEN-00312-000', 'BraTS-MEN-01074-000', 'BraTS-MEN-01353-000', 'BraTS-MEN-00877-000', 
                'BraTS-MEN-00702-000', 'BraTS-MEN-00819-000', 'BraTS-MEN-00219-000', 'BraTS-MEN-00045-000', 
                'BraTS-MEN-01421-000', 'BraTS-MEN-01085-000', 'BraTS-MEN-00818-000', 'BraTS-MEN-00462-000', 
                'BraTS-MEN-00074-003', 'BraTS-MEN-00779-000', 'BraTS-MEN-00810-000', 'BraTS-MEN-00516-000', 
                'BraTS-MEN-00868-000', 'BraTS-MEN-01112-000', 'BraTS-MEN-00255-000', 'BraTS-MEN-00777-000', 
                'BraTS-MEN-00493-000', 'BraTS-MEN-01333-000', 'BraTS-MEN-01419-000', 'BraTS-MEN-00505-000', 
                'BraTS-MEN-00744-000', 'BraTS-MEN-00909-000', 'BraTS-MEN-00103-000', 'BraTS-MEN-01157-000', 
                'BraTS-MEN-01354-000', 'BraTS-MEN-01094-000', 'BraTS-MEN-01044-000', 'BraTS-MEN-01061-000', 
                'BraTS-MEN-01266-000', 'BraTS-MEN-00704-000', 'BraTS-MEN-00267-000', 'BraTS-MEN-00711-000', 
                'BraTS-MEN-01141-000', 'BraTS-MEN-00212-000', 'BraTS-MEN-00542-000', 'BraTS-MEN-00053-000', 
                'BraTS-MEN-01432-000', 'BraTS-MEN-00078-000', 'BraTS-MEN-00970-000', 'BraTS-MEN-00599-000', 
                'BraTS-MEN-01215-000', 'BraTS-MEN-00918-000', 'BraTS-MEN-00627-000', 'BraTS-MEN-00417-000', 
                'BraTS-MEN-00187-000', 'BraTS-MEN-00534-000', 'BraTS-MEN-00414-000', 'BraTS-MEN-01203-000', 
                'BraTS-MEN-00504-000', 'BraTS-MEN-00947-000', 'BraTS-MEN-00132-000', 'BraTS-MEN-00175-000', 
                'BraTS-MEN-01371-000', 'BraTS-MEN-00074-000', 'BraTS-MEN-00563-000', 'BraTS-MEN-00932-003', 
                'BraTS-MEN-01315-000', 'BraTS-MEN-00925-000', 'BraTS-MEN-00173-000', 'BraTS-MEN-00690-000', 
                'BraTS-MEN-00587-000', 'BraTS-MEN-01404-000', 'BraTS-MEN-00429-000', 'BraTS-MEN-01435-001', 
                'BraTS-MEN-01134-000', 'BraTS-MEN-00338-000', 'BraTS-MEN-01224-000', 'BraTS-MEN-01336-000', 
                'BraTS-MEN-00558-000', 'BraTS-MEN-00466-000', 'BraTS-MEN-00075-000', 'BraTS-MEN-00959-000', 
                'BraTS-MEN-00644-000', 'BraTS-MEN-00604-000', 'BraTS-MEN-01170-000', 'BraTS-MEN-00665-000', 
                'BraTS-MEN-01055-000', 'BraTS-MEN-00935-000', 'BraTS-MEN-01005-000', 'BraTS-MEN-00693-000', 
                'BraTS-MEN-00146-000', 'BraTS-MEN-00974-000', 'BraTS-MEN-00172-000', 'BraTS-MEN-00404-000', 
                'BraTS-MEN-00040-000', 'BraTS-MEN-00138-000', 'BraTS-MEN-01063-000', 'BraTS-MEN-00741-000', 
                'BraTS-MEN-00166-000', 'BraTS-MEN-01010-000', 'BraTS-MEN-00467-000', 'BraTS-MEN-01114-000', 
                'BraTS-MEN-00890-000', 'BraTS-MEN-00213-000', 'BraTS-MEN-01399-000', 'BraTS-MEN-00334-000', 
                'BraTS-MEN-00327-000', 'BraTS-MEN-00553-000', 'BraTS-MEN-01047-000', 'BraTS-MEN-01250-000', 
                'BraTS-MEN-00932-009', 'BraTS-MEN-01198-004', 'BraTS-MEN-00655-000', 'BraTS-MEN-00514-000', 
                'BraTS-MEN-00024-000', 'BraTS-MEN-00645-000', 'BraTS-MEN-00743-001', 'BraTS-MEN-01312-000', 
                'BraTS-MEN-00375-000', 'BraTS-MEN-01200-000', 'BraTS-MEN-00751-000', 'BraTS-MEN-00086-000', 
                'BraTS-MEN-01070-000', 'BraTS-MEN-00552-000', 'BraTS-MEN-01065-000', 'BraTS-MEN-00083-000', 
                'BraTS-MEN-00319-000', 'BraTS-MEN-00795-000', 'BraTS-MEN-01119-000', 'BraTS-MEN-00811-000', 
                'BraTS-MEN-00425-000', 'BraTS-MEN-00112-000', 'BraTS-MEN-01259-000', 'BraTS-MEN-01359-000', 
                'BraTS-MEN-00047-000', 'BraTS-MEN-00829-000', 'BraTS-MEN-00506-000', 'BraTS-MEN-00535-000', 
                'BraTS-MEN-00578-000', 'BraTS-MEN-00691-000', 'BraTS-MEN-00876-000', 'BraTS-MEN-01144-000', 
                'BraTS-MEN-01368-000', 'BraTS-MEN-00686-000', 'BraTS-MEN-00459-000', 'BraTS-MEN-01039-000', 
                'BraTS-MEN-01009-000', 'BraTS-MEN-00141-000', 'BraTS-MEN-01008-012', 'BraTS-MEN-00653-000', 
                'BraTS-MEN-00590-000', 'BraTS-MEN-00887-000', 'BraTS-MEN-00832-000', 'BraTS-MEN-00278-000', 
                'BraTS-MEN-00133-000', 'BraTS-MEN-00621-000', 'BraTS-MEN-00574-000', 'BraTS-MEN-00359-000', 
                'BraTS-MEN-00858-000', 'BraTS-MEN-00566-000', 'BraTS-MEN-01067-000', 'BraTS-MEN-01076-000', 
                'BraTS-MEN-01246-001', 'BraTS-MEN-00885-000', 'BraTS-MEN-01357-000', 'BraTS-MEN-00510-000', 
                'BraTS-MEN-00161-000', 'BraTS-MEN-01060-000', 'BraTS-MEN-00164-000', 'BraTS-MEN-01092-000', 
                'BraTS-MEN-00423-000', 'BraTS-MEN-00288-000', 'BraTS-MEN-00805-000', 'BraTS-MEN-00017-000', 
                'BraTS-MEN-01226-000', 'BraTS-MEN-00270-000', 'BraTS-MEN-00268-000', 'BraTS-MEN-00431-000', 
                'BraTS-MEN-00967-000', 'BraTS-MEN-01291-000', 'BraTS-MEN-00854-000', 'BraTS-MEN-01416-000', 
                'BraTS-MEN-01079-000', 'BraTS-MEN-00198-000', 'BraTS-MEN-00247-000', 'BraTS-MEN-00393-000', 
                'BraTS-MEN-00774-000', 'BraTS-MEN-00895-000', 'BraTS-MEN-01198-011', 'BraTS-MEN-00993-000', 
                'BraTS-MEN-00991-000', 'BraTS-MEN-00755-000', 'BraTS-MEN-01351-000', 'BraTS-MEN-00074-008', 
                'BraTS-MEN-00984-000', 'BraTS-MEN-01407-000', 'BraTS-MEN-01424-000', 'BraTS-MEN-00666-000', 
                'BraTS-MEN-00719-000', 'BraTS-MEN-00438-000', 'BraTS-MEN-00620-000', 'BraTS-MEN-00108-005', 
                'BraTS-MEN-00084-000', 'BraTS-MEN-00233-000', 'BraTS-MEN-01136-000', 'BraTS-MEN-01107-000', 
                'BraTS-MEN-01026-000', 'BraTS-MEN-00983-008', 'BraTS-MEN-01187-000', 'BraTS-MEN-00759-000', 
                'BraTS-MEN-00168-000', 'BraTS-MEN-00443-000', 'BraTS-MEN-00596-001', 'BraTS-MEN-00717-010', 
                'BraTS-MEN-01046-000', 'BraTS-MEN-00596-000', 'BraTS-MEN-01167-000', 'BraTS-MEN-00717-012', 
                'BraTS-MEN-00921-000', 'BraTS-MEN-00797-000', 'BraTS-MEN-01008-009', 'BraTS-MEN-00362-000', 
                'BraTS-MEN-01396-000', 'BraTS-MEN-00648-000', 'BraTS-MEN-01433-000', 'BraTS-MEN-00311-000', 
                'BraTS-MEN-00904-000', 'BraTS-MEN-00150-000', 'BraTS-MEN-00626-000', 'BraTS-MEN-00781-009', 
                'BraTS-MEN-01329-000', 'BraTS-MEN-00750-000', 'BraTS-MEN-00576-007', 'BraTS-MEN-00227-000', 
                'BraTS-MEN-01038-000', 'BraTS-MEN-01153-000', 'BraTS-MEN-01090-000', 'BraTS-MEN-00260-000', 
                'BraTS-MEN-01330-000', 'BraTS-MEN-00482-000', 'BraTS-MEN-00370-000', 'BraTS-MEN-00705-000', 
                'BraTS-MEN-00054-000', 'BraTS-MEN-00081-000', 'BraTS-MEN-01158-000', 'BraTS-MEN-00989-000', 
                'BraTS-MEN-00154-000', 'BraTS-MEN-00269-000', 'BraTS-MEN-01321-000', 'BraTS-MEN-00920-000', 
                'BraTS-MEN-01189-000', 'BraTS-MEN-00833-000', 'BraTS-MEN-00896-000', 'BraTS-MEN-00787-000', 
                'BraTS-MEN-00487-000', 'BraTS-MEN-00945-000', 'BraTS-MEN-01300-000', 'BraTS-MEN-00971-000', 
                'BraTS-MEN-00562-000', 'BraTS-MEN-01198-013', 'BraTS-MEN-00957-000', 'BraTS-MEN-00043-000', 
                'BraTS-MEN-00739-000', 'BraTS-MEN-00306-000', 'BraTS-MEN-00369-000', 'BraTS-MEN-00481-000', 
                'BraTS-MEN-00770-000', 'BraTS-MEN-01151-000', 'BraTS-MEN-00276-000', 'BraTS-MEN-01383-007', 
                'BraTS-MEN-00544-000', 'BraTS-MEN-00523-000', 'BraTS-MEN-01349-000', 'BraTS-MEN-00973-000', 
                'BraTS-MEN-00683-000', 'BraTS-MEN-00388-000', 'BraTS-MEN-00825-000', 'BraTS-MEN-00613-000', 
                'BraTS-MEN-01298-000', 'BraTS-MEN-00717-007', 'BraTS-MEN-01426-000', 'BraTS-MEN-00259-000', 
                'BraTS-MEN-00564-000', 'BraTS-MEN-00221-000', 'BraTS-MEN-01126-000', 'BraTS-MEN-00758-000', 
                'BraTS-MEN-00503-000', 'BraTS-MEN-00922-000', 'BraTS-MEN-00056-000', 'BraTS-MEN-00625-000', 
                'BraTS-MEN-00763-000', 'BraTS-MEN-01105-000', 'BraTS-MEN-00232-000', 'BraTS-MEN-00206-000', 
                'BraTS-MEN-00389-000', 'BraTS-MEN-00685-000', 'BraTS-MEN-00765-000', 'BraTS-MEN-00199-000', 
                'BraTS-MEN-00441-000', 'BraTS-MEN-00400-000', 'BraTS-MEN-01414-000', 'BraTS-MEN-01123-000', 
                'BraTS-MEN-01122-000', 'BraTS-MEN-00965-000', 'BraTS-MEN-00821-000', 'BraTS-MEN-00023-000', 
                'BraTS-MEN-01163-000', 'BraTS-MEN-00837-000', 'BraTS-MEN-00476-000', 'BraTS-MEN-00307-000', 
                'BraTS-MEN-00608-000', 'BraTS-MEN-00381-000', 'BraTS-MEN-00332-000', 'BraTS-MEN-00820-000', 
                'BraTS-MEN-00418-000', 'BraTS-MEN-00224-000', 'BraTS-MEN-00163-000', 'BraTS-MEN-00579-000', 
                'BraTS-MEN-00936-000', 'BraTS-MEN-00696-000', 'BraTS-MEN-01036-000', 'BraTS-MEN-00471-000', 
                'BraTS-MEN-00764-000', 'BraTS-MEN-00898-000', 'BraTS-MEN-00823-000', 'BraTS-MEN-00923-000', 
                'BraTS-MEN-01290-000', 'BraTS-MEN-01018-000', 'BraTS-MEN-00444-000', 'BraTS-MEN-00032-000', 
                'BraTS-MEN-00398-000', 'BraTS-MEN-00415-000', 'BraTS-MEN-00932-006', 'BraTS-MEN-00399-000', 
                'BraTS-MEN-00293-000', 'BraTS-MEN-00364-000', 'BraTS-MEN-00511-000', 'BraTS-MEN-00110-000', 
                'BraTS-MEN-01050-000', 'BraTS-MEN-01002-000', 'BraTS-MEN-00983-006', 'BraTS-MEN-00356-000', 
                'BraTS-MEN-00638-000', 'BraTS-MEN-00583-000', 'BraTS-MEN-00020-000', 'BraTS-MEN-00846-000', 
                'BraTS-MEN-00663-000', 'BraTS-MEN-00096-000', 'BraTS-MEN-00174-000', 'BraTS-MEN-01350-000', 
                'BraTS-MEN-00037-000', 'BraTS-MEN-01362-000', 'BraTS-MEN-00405-000', 'BraTS-MEN-01081-000', 
                'BraTS-MEN-00612-000', 'BraTS-MEN-01237-000', 'BraTS-MEN-00371-000', 'BraTS-MEN-00616-000', 
                'BraTS-MEN-00697-000', 'BraTS-MEN-00762-000', 'BraTS-MEN-01383-006', 'BraTS-MEN-00878-000', 
                'BraTS-MEN-01342-000', 'BraTS-MEN-00637-000', 'BraTS-MEN-00113-000', 'BraTS-MEN-00603-000', 
                'BraTS-MEN-01089-000', 'BraTS-MEN-00200-000', 'BraTS-MEN-01367-000', 'BraTS-MEN-01284-000', 
                'BraTS-MEN-00410-000', 'BraTS-MEN-00033-000', 'BraTS-MEN-00245-000', 'BraTS-MEN-00622-000', 
                'BraTS-MEN-01059-000', 'BraTS-MEN-00676-000', 'BraTS-MEN-00576-006', 'BraTS-MEN-00549-000', 
                'BraTS-MEN-01156-000', 'BraTS-MEN-00016-000', 'BraTS-MEN-00170-000', 'BraTS-MEN-00698-000', 
                'BraTS-MEN-00478-000', 'BraTS-MEN-00873-000', 'BraTS-MEN-01077-000', 'BraTS-MEN-00662-000', 
                'BraTS-MEN-01214-000', 'BraTS-MEN-00498-000', 'BraTS-MEN-01392-000', 'BraTS-MEN-00396-000', 
                'BraTS-MEN-00631-000', 'BraTS-MEN-01058-000', 'BraTS-MEN-00669-000', 'BraTS-MEN-01000-000', 
                'BraTS-MEN-00527-000', 'BraTS-MEN-00390-000', 'BraTS-MEN-01073-000', 'BraTS-MEN-00664-000', 
                'BraTS-MEN-00336-000', 'BraTS-MEN-00529-000', 'BraTS-MEN-00501-000', 'BraTS-MEN-00321-000', 
                'BraTS-MEN-01403-000', 'BraTS-MEN-01135-000', 'BraTS-MEN-01316-000', 'BraTS-MEN-00468-000', 
                'BraTS-MEN-00076-000', 'BraTS-MEN-00111-000', 'BraTS-MEN-00628-000', 'BraTS-MEN-00424-000', 
                'BraTS-MEN-00285-000', 'BraTS-MEN-01129-000', 'BraTS-MEN-00363-000', 'BraTS-MEN-00782-000', 
                'BraTS-MEN-00165-000', 'BraTS-MEN-01148-000', 'BraTS-MEN-00345-000', 'BraTS-MEN-01078-000', 
                'BraTS-MEN-01198-007', 'BraTS-MEN-00932-008', 'BraTS-MEN-00610-000', 'BraTS-MEN-01017-000', 
                'BraTS-MEN-01327-000', 'BraTS-MEN-00320-000', 'BraTS-MEN-00681-000', 'BraTS-MEN-00052-000', 'BraTS-MEN-01033-000', 'BraTS-MEN-00835-000', 'BraTS-MEN-01008-010', 'BraTS-MEN-00999-001', 
                'BraTS-MEN-00650-000', 'BraTS-MEN-00790-000', 'BraTS-MEN-00409-000', 'BraTS-MEN-01254-000', 'BraTS-MEN-00573-000', 'BraTS-MEN-00948-000', 'BraTS-MEN-00639-000', 'BraTS-MEN-00346-000', 
                'BraTS-MEN-00725-000', 'BraTS-MEN-00674-000', 'BraTS-MEN-00844-000', 'BraTS-MEN-00768-000', 'BraTS-MEN-00606-000', 'BraTS-MEN-00932-007', 'BraTS-MEN-00021-000', 'BraTS-MEN-01252-000', 
                'BraTS-MEN-01083-000', 'BraTS-MEN-01093-000', 'BraTS-MEN-00313-000', 'BraTS-MEN-00131-000', 'BraTS-MEN-00101-000', 'BraTS-MEN-00136-000', 'BraTS-MEN-00880-000', 'BraTS-MEN-00258-000', 
                'BraTS-MEN-00237-000', 'BraTS-MEN-01020-000', 'BraTS-MEN-00897-000', 'BraTS-MEN-00066-000', 'BraTS-MEN-01275-000', 'BraTS-MEN-00738-000', 'BraTS-MEN-01176-000', 'BraTS-MEN-01108-000', 
                'BraTS-MEN-01132-000', 'BraTS-MEN-00533-000', 'BraTS-MEN-00861-000', 'BraTS-MEN-00712-000', 'BraTS-MEN-00717-008', 'BraTS-MEN-01383-000', 'BraTS-MEN-01283-000', 'BraTS-MEN-01048-000', 
                'BraTS-MEN-00470-000', 'BraTS-MEN-00512-000', 'BraTS-MEN-00615-000', 'BraTS-MEN-00930-000', 'BraTS-MEN-00106-000', 'BraTS-MEN-00773-000', 'BraTS-MEN-00281-000', 'BraTS-MEN-00484-000', 
                'BraTS-MEN-00074-007', 'BraTS-MEN-00996-000', 'BraTS-MEN-00717-005', 'BraTS-MEN-00932-004', 'BraTS-MEN-00461-000', 'BraTS-MEN-00202-000', 'BraTS-MEN-01185-000', 'BraTS-MEN-00607-000', 
                'BraTS-MEN-00559-000', 'BraTS-MEN-00451-000', 'BraTS-MEN-01221-000', 'BraTS-MEN-00253-000', 'BraTS-MEN-00420-000', 'BraTS-MEN-00071-000', 'BraTS-MEN-00062-000', 'BraTS-MEN-00525-000', 
                'BraTS-MEN-01355-000', 'BraTS-MEN-00572-000', 'BraTS-MEN-00524-000', 'BraTS-MEN-01375-000', 'BraTS-MEN-01164-000', 'BraTS-MEN-00231-000', 'BraTS-MEN-00215-000', 'BraTS-MEN-00022-000'])
            splits[self.fold]['val'] = np.array(
                ['BraTS-MEN-01280-000', 'BraTS-MEN-01171-000', 'BraTS-MEN-00576-009', 'BraTS-MEN-00330-000', 'BraTS-MEN-00279-000', 'BraTS-MEN-00789-000', 'BraTS-MEN-00576-005', 'BraTS-MEN-00658-000', 
                'BraTS-MEN-00688-000', 'BraTS-MEN-00963-000', 'BraTS-MEN-01003-007', 'BraTS-MEN-00983-009', 'BraTS-MEN-00571-000', 'BraTS-MEN-00495-000', 'BraTS-MEN-00824-000', 'BraTS-MEN-01234-000', 
                'BraTS-MEN-00326-000', 'BraTS-MEN-01431-000', 'BraTS-MEN-00073-000', 'BraTS-MEN-00034-000', 'BraTS-MEN-00433-000', 'BraTS-MEN-00236-000', 'BraTS-MEN-00176-000', 'BraTS-MEN-00541-000', 
                'BraTS-MEN-01429-000', 'BraTS-MEN-00208-000', 'BraTS-MEN-00403-000', 'BraTS-MEN-01121-000', 'BraTS-MEN-00682-000', 'BraTS-MEN-01192-000', 'BraTS-MEN-01117-000', 'BraTS-MEN-00123-000', 
                'BraTS-MEN-00609-000', 'BraTS-MEN-01231-000', 'BraTS-MEN-00157-000', 'BraTS-MEN-00286-000', 'BraTS-MEN-00434-000', 'BraTS-MEN-00474-000', 'BraTS-MEN-00315-000', 'BraTS-MEN-00907-000', 
                'BraTS-MEN-00181-000', 'BraTS-MEN-01411-000', 'BraTS-MEN-00155-000', 'BraTS-MEN-01323-000', 'BraTS-MEN-00781-005', 'BraTS-MEN-00010-000', 'BraTS-MEN-01339-000', 'BraTS-MEN-00341-000', 
                'BraTS-MEN-00012-000', 'BraTS-MEN-01360-000', 'BraTS-MEN-00632-000', 'BraTS-MEN-00983-010', 'BraTS-MEN-00028-000', 'BraTS-MEN-01335-000', 'BraTS-MEN-00129-000', 'BraTS-MEN-01370-000', 
                'BraTS-MEN-01340-000', 'BraTS-MEN-00299-000', 'BraTS-MEN-00630-000', 'BraTS-MEN-00074-009', 'BraTS-MEN-01080-000', 'BraTS-MEN-00491-000', 'BraTS-MEN-01314-000', 'BraTS-MEN-00734-000', 
                'BraTS-MEN-01220-000', 'BraTS-MEN-00004-000', 'BraTS-MEN-00903-000', 'BraTS-MEN-01030-000', 'BraTS-MEN-01172-000', 'BraTS-MEN-00780-000', 'BraTS-MEN-00672-000', 'BraTS-MEN-00239-000', 
                'BraTS-MEN-01430-000', 'BraTS-MEN-00094-000', 'BraTS-MEN-00671-000', 'BraTS-MEN-00256-000', 'BraTS-MEN-00074-005', 'BraTS-MEN-00633-000', 'BraTS-MEN-00987-000', 'BraTS-MEN-00614-000', 
                'BraTS-MEN-00057-000', 'BraTS-MEN-01434-000', 'BraTS-MEN-01307-000', 'BraTS-MEN-00580-000', 'BraTS-MEN-01296-000', 'BraTS-MEN-00946-000', 'BraTS-MEN-00354-000', 'BraTS-MEN-01014-000', 
                'BraTS-MEN-00244-000', 'BraTS-MEN-00520-000', 'BraTS-MEN-00305-000', 'BraTS-MEN-00889-000', 'BraTS-MEN-01326-000', 'BraTS-MEN-01218-000', 'BraTS-MEN-00966-000', 'BraTS-MEN-00629-000', 
                'BraTS-MEN-01271-000', 'BraTS-MEN-01383-005', 'BraTS-MEN-00207-000', 'BraTS-MEN-01099-000', 'BraTS-MEN-00940-000', 'BraTS-MEN-00884-000', 'BraTS-MEN-01071-000', 'BraTS-MEN-00999-000', 
                'BraTS-MEN-01261-000', 'BraTS-MEN-00910-000', 'BraTS-MEN-00548-000', 'BraTS-MEN-00717-009', 'BraTS-MEN-00380-000', 'BraTS-MEN-00210-000', 'BraTS-MEN-00435-000', 'BraTS-MEN-01101-000', 
                'BraTS-MEN-01143-000', 'BraTS-MEN-01169-000', 'BraTS-MEN-00114-000', 'BraTS-MEN-00687-000', 'BraTS-MEN-00025-000', 'BraTS-MEN-01012-000', 'BraTS-MEN-00828-000', 'BraTS-MEN-00235-000', 
                'BraTS-MEN-01053-000', 'BraTS-MEN-00981-000', 'BraTS-MEN-00598-000', 'BraTS-MEN-01181-000', 'BraTS-MEN-00964-000', 'BraTS-MEN-00490-000', 'BraTS-MEN-01285-000', 'BraTS-MEN-00841-000', 
                'BraTS-MEN-00623-000', 'BraTS-MEN-01003-010', 'BraTS-MEN-00708-000', 'BraTS-MEN-00026-000', 'BraTS-MEN-00271-000', 'BraTS-MEN-00985-000', 'BraTS-MEN-00667-000', 'BraTS-MEN-00894-000', 
                'BraTS-MEN-00179-000', 'BraTS-MEN-01389-000', 'BraTS-MEN-00781-007', 'BraTS-MEN-00532-000', 'BraTS-MEN-00246-000', 'BraTS-MEN-00243-000', 'BraTS-MEN-01001-000', 'BraTS-MEN-00085-000', 
                'BraTS-MEN-01320-000', 'BraTS-MEN-00997-000', 'BraTS-MEN-01147-000', 'BraTS-MEN-00263-000', 'BraTS-MEN-00842-000', 'BraTS-MEN-00706-000', 'BraTS-MEN-01165-000', 'BraTS-MEN-00557-000', 
                'BraTS-MEN-01095-000', 'BraTS-MEN-00318-000', 'BraTS-MEN-01124-000', 'BraTS-MEN-01216-000', 'BraTS-MEN-01096-000', 'BraTS-MEN-01256-000', 'BraTS-MEN-00069-000', 'BraTS-MEN-00703-000', 
                'BraTS-MEN-00746-000', 'BraTS-MEN-01104-000', 'BraTS-MEN-00284-000', 'BraTS-MEN-00659-000', 'BraTS-MEN-00545-000', 'BraTS-MEN-01282-000', 'BraTS-MEN-00962-000', 'BraTS-MEN-00108-004', 
                'BraTS-MEN-00352-000', 'BraTS-MEN-00105-000', 'BraTS-MEN-01233-000', 'BraTS-MEN-01152-000', 'BraTS-MEN-00927-000', 'BraTS-MEN-00641-000', 'BraTS-MEN-00804-000', 'BraTS-MEN-00042-000', 
                'BraTS-MEN-01409-000', 'BraTS-MEN-00088-000', 'BraTS-MEN-01062-000', 'BraTS-MEN-00335-000', 'BraTS-MEN-00717-006', 'BraTS-MEN-00194-000', 'BraTS-MEN-01051-000', 'BraTS-MEN-00290-000', 
                'BraTS-MEN-00308-000', 'BraTS-MEN-00041-000', 'BraTS-MEN-01128-000', 'BraTS-MEN-00065-000', 'BraTS-MEN-01383-010', 'BraTS-MEN-00575-000', 'BraTS-MEN-01257-000', 'BraTS-MEN-01206-000', 
                'BraTS-MEN-00183-000', 'BraTS-MEN-01247-000', 'BraTS-MEN-00905-000', 'BraTS-MEN-01242-000', 'BraTS-MEN-00942-000', 'BraTS-MEN-01279-002', 'BraTS-MEN-00048-000', 'BraTS-MEN-01380-000'])

            # split
            split_number = round(len(splits[self.fold]['train']) * (self.split / 100))
            print('split percentage:', self.split / 100)
            save_pickle(splits, splits_file)
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
