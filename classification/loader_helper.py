'''The following module deals with creating the loader he'''
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from data_declaration import MRIDataset, Task
from data_declaration import ToTensor


class LoaderHelper:
    '''An abstract class for assisting with dataset creation.'''
    def __init__(self, task: Task = Task.NC_v_AD):

        self.task = task
        self.labels = []

        if task == Task.NC_v_AD:
            self.labels = ["NC", "AD"]
        else:
            self.labels = ["sMCI", "pMCI"]

        #cat12
        self.train_ds = MRIDataset(root_dir="./datasets/adni1/",
                labels=self.labels,
                training=True,
                transform=transforms.Compose([
                ToTensor()
            ]))
        self.test_ds = MRIDataset(root_dir="./datasets/adni2/",
                labels=self.labels,
                training=False,
                transform=transforms.Compose([
                ToTensor()
            ]))



        #print(self.train_ds.len)
        self.indices = []
        self.set_indices()
        


    def get_task(self):
        '''gets task'''
        return self.task


    def get_task_string(self):
        '''Gets task string'''
        if self.task == Task.NC_v_AD:
            return "NC_v_AD"
        else:
            return "sMCI_v_pMCI"


    def change_ds_labels(self, labels_in):
        '''Function to change the labels of the dataset obj.'''
        self.dataset = MRIDataset(root_dir="../data/",
                                  labels=labels_in,
                                  transform=transforms.Compose([
                                      ToTensor()])
                                 )


    def change_task(self, task: Task):
        '''Function to change task of the Datasets'''
        self.task = task
        
        if (task == Task.NC_v_AD):
            self.labels = ["NC", "AD"]
        else:
            self.labels = ["sMCI", "pMCI"]

        self.dataset = MRIDataset(root_dir="../data/",
                            labels=self.labels,
                            transform=transforms.Compose([
                                ToTensor()])
                            )

        self.set_indices()


    def set_indices(self, total_folds=5):
        '''Abstract function to set indices'''
        # test_split = .2
        # shuffle_dataset = True
        # random_seed = 42

        # dataset_size = len(self.dataset)
        # indices = list(range(dataset_size))
        # split = int(np.floor(test_split * dataset_size))

        # if shuffle_dataset:
        #     np.random.seed(random_seed)
        #     np.random.shuffle(indices)

        # fold_indices = []
        # lb_split = 0
        # ub_split = split

        # for _ in range(total_folds):
        #     train_indices = indices[:lb_split] + indices[ub_split:]
        #     test_indices = indices[lb_split:ub_split]
        #     lb_split = split
        #     ub_split = 2*split #only works if kfold is 5 so be carefull
        #     fold_indices.append((train_indices, test_indices))
        shuffle_dataset = True
        random_seed = 42
        train_dataset_size = len(self.train_ds)
        test_dataset_size = len(self.test_ds)
        train_indices = list(range(train_dataset_size))
        test_indices = list(range(test_dataset_size))        
        
        self.indices = [train_indices, test_indices]


    def make_loaders(self, shuffle=True):
        '''Makes the loaders'''
        fold_indices = self.indices()

        for k in range(5):

            train_ds = Subset(self.dataset, fold_indices[k][0])
            test_ds  = Subset(self.dataset, fold_indices[k][1])

            train_dl = DataLoader(train_ds, batch_size=2, shuffle=shuffle, num_workers=4, drop_last=True)
            test_dl = DataLoader(test_ds,  batch_size=2, shuffle=shuffle, num_workers=4, drop_last=True)

        print(len(test_ds))

        return (train_dl, test_dl)

    
    def get_train_dl(self, fold_ind, shuffle=True):

        
        train_ds = Subset(self.train_ds, self.indices[0])
        train_dl = DataLoader(train_ds, batch_size=2, shuffle=shuffle, num_workers=4, drop_last=True)

        return train_dl


    def get_test_dl(self, fold_ind, shuffle=True):

        test_ds = Subset(self.test_ds, self.indices[1])
        test_dl = DataLoader(test_ds, batch_size=2, shuffle=shuffle, num_workers=4, drop_last=True)

        return test_dl



