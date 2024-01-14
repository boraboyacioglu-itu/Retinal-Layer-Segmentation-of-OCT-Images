import os
import cv2
import torch
import numpy as np
import datetime

# Additional Scripts
from train_transunet import TransUNetSeg
from utils_octa.utils_octa import thresh_func
from config_octa import cfg


class SegInference:
    def __init__(self, model_path, device):
        self.device = device
        self.transunet = TransUNetSeg(device)
        self.transunet.load_model(model_path)

        if not os.path.exists('./results'):
            os.mkdir('./results')

    def read_and_preprocess(self, p):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_torch = cv2.resize(img, (cfg.transunet.img_dim, cfg.transunet.img_dim))
        img_torch = img_torch / 255.
        img_torch = img_torch.transpose((2, 0, 1))
        img_torch = np.expand_dims(img_torch, axis=0)
        img_torch = torch.from_numpy(img_torch.astype('float32')).to(self.device)

        return img, img_torch

    def save_preds(self, preds):
        folder_path = './results/' + str(datetime.datetime.utcnow()).replace(' ', '_')
        os.mkdir(folder_path)

        for name, masks in preds.items():
            for i, mask in enumerate(masks):
                cv2.imwrite(f'{folder_path}/{name}_class_{i}.png', mask)

    def infer(self, path, save=True):
        path = [path] if isinstance(path, str) else path

        preds = {}
        for p in path:
            file_name = p.split('/')[-1]
            img, img_torch = self.read_and_preprocess(p)
            with torch.no_grad():
                pred_masks = self.transunet.model(img_torch)
                pred_masks = torch.sigmoid(pred_masks)
                pred_masks = pred_masks.detach().cpu().numpy().transpose((0, 2, 3, 1))

            orig_h, orig_w = img.shape[:2]
            class_masks = []
            for i in range(cfg.transunet.class_num):  # Assuming class_num is the number of classes
                pred_mask = cv2.resize(pred_masks[0, ..., i], (orig_w, orig_h))
                pred_mask = thresh_func(pred_mask, thresh=cfg.inference_threshold)
                pred_mask *= 255
                class_masks.append(pred_mask.astype('uint8'))

            preds[file_name] = class_masks

        if save:
            self.save_preds(preds)

        return preds