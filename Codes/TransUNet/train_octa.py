from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from utils_octa.octdataset import OCTDataset
from utils_octa.utils_octa import EpochCallback

from config_octa import cfg

from train_transunet import TransUNetSeg


class TrainTestPipe:
    def __init__(self, train_path, test_path, model_path, device, cfg, log_file=None):
        self.device = device
        self.model_path = model_path

        self.train_loader = self.__load_dataset(train_path, train=True)
        self.test_loader = self.__load_dataset(test_path)

        self.transunet = TransUNetSeg(self.device)
        self.log_file = log_file
    def __load_dataset(self, path, train=False):
        shuffle = False
        transform = False
        set = OCTDataset(path, transform)
        loader = DataLoader(set, batch_size=cfg.batch_size, shuffle=shuffle)

        return loader

    def __loop(self, loader, step_func, t):
        total_loss = 0

        for step, data in enumerate(loader):
            img, mask = data['img'], data['mask']
            img = img.to(self.device)
            mask = mask.to(self.device)

            loss = step_func(img=img, mask=mask)

            total_loss += loss
            t.update()

        return total_loss

    @staticmethod
    def format_cfg_params(cfg):
        def recurse_format(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f'{parent_key}.{k}' if parent_key else k
                if isinstance(v, dict):
                    items.extend(recurse_format(v, new_key))
                else:
                    items.append(f'{new_key}: {v}')
            return items

        return ', '.join(recurse_format(cfg))

    def train(self):
        callback = EpochCallback(self.model_path, cfg.epoch,
                                 self.transunet.model, self.transunet.optimizer, 'test_loss', cfg.patience, self.log_file)
        for epoch in range(cfg.epoch):
            with tqdm(total=len(self.train_loader) + len(self.test_loader)) as t:
                train_loss = self.__loop(self.train_loader, self.transunet.train_step, t)
                test_loss = self.__loop(self.test_loader, self.transunet.test_step, t)
                
            callback.epoch_end(epoch + 1,
                               {'loss': train_loss / len(self.train_loader),
                                'test_loss': test_loss / len(self.test_loader)})

            if callback.end_training:
                break
                
        # Log cfg parameters and best model performance
        if self.log_file:
            print("burda")
            cfg_params_str = self.format_cfg_params(cfg)
            self.log_file.write(f'Best model with performance {callback.best_performance} and params {cfg_params_str}\n')