import torch
import numpy as np
from config_octa import cfg

def thresh_func(mask, thresh=0.5):
    mask[mask >= thresh] = 1
    mask[mask < thresh] = 0

    return mask

# dice loss for multiple layers
def dice_loss(pred, target, num_layers):
    pred = torch.sigmoid(pred)
    
    total_loss = 0
    for layer in range(num_layers):
        pred_layer = pred[:, layer, :, :].contiguous().view(-1)
        target_layer = target[:, layer, :, :].contiguous().view(-1)

        intersection = torch.sum(pred_layer * target_layer)
        pred_sum = torch.sum(pred_layer * pred_layer)
        target_sum = torch.sum(target_layer * target_layer)

        layer_loss = 1 - ((2. * intersection + 1e-5) / (pred_sum + target_sum + 1e-5))
        total_loss += layer_loss

    return total_loss / num_layers

def precision_score_(groundtruth_masks, pred_masks, num_layers):
    precision_scores = []
    for layer in range(num_layers):
        pred_mask = pred_masks[:, layer, :, :]
        groundtruth_mask = groundtruth_masks[:, layer, :, :]

        intersect = torch.sum(pred_mask * groundtruth_mask)
        total_pixel_pred = torch.sum(pred_mask)

        precision = intersect.item() / total_pixel_pred.item() if total_pixel_pred > 0 else 0
        precision_scores.append(round(precision, 3))

    return precision_scores

def recall_score_(groundtruth_masks, pred_masks, num_layers):
    recall_scores = []
    for layer in range(num_layers):
        pred_mask = pred_masks[:, layer, :, :]
        groundtruth_mask = groundtruth_masks[:, layer, :, :]

        intersect = torch.sum(pred_mask * groundtruth_mask)
        total_pixel_truth = torch.sum(groundtruth_mask)

        recall = intersect / total_pixel_truth if total_pixel_truth > 0 else 0
        recall_scores.append(round(recall.item(), 3))

    return recall_scores


def accuracy_(groundtruth_masks, pred_masks, num_layers):
    accuracies = []
    for layer in range(num_layers):
        pred_mask = pred_masks[:, layer, :, :]
        groundtruth_mask = groundtruth_masks[:, layer, :, :]
        intersect = torch.sum(pred_mask * groundtruth_mask)
        union = torch.sum(pred_mask) + torch.sum(groundtruth_mask) - intersect
        xor = torch.sum(groundtruth_mask == pred_mask)
        acc = xor / (union + xor - intersect) if (union + xor - intersect) > 0 else 0
        accuracies.append(round(acc.item(), 3))
    return accuracies

def dice_coef(groundtruth_masks, pred_masks, num_layers):
    dice_scores = []
    for layer in range(num_layers):
        intersect = torch.sum(pred_masks[:, layer, :, :] * groundtruth_masks[:, layer, :, :])
        total_sum = torch.sum(pred_masks[:, layer, :, :]) + torch.sum(groundtruth_masks[:, layer, :, :])
        dice = 2 * intersect / total_sum if total_sum > 0 else 0
        dice_scores.append(round(dice.item(), 3))
    return dice_scores


def iou(groundtruth_masks, pred_masks, num_layers):
    iou_scores = []
    for layer in range(num_layers):
        pred_mask = pred_masks[:, layer, :, :]
        groundtruth_mask = groundtruth_masks[:, layer, :, :]
        intersect = torch.sum(pred_mask * groundtruth_mask)
        union = torch.sum(pred_mask) + torch.sum(groundtruth_mask) - intersect
        iou = intersect / union if union > 0 else 0
        iou_scores.append(round(iou.item(), 3))
    return iou_scores

class EpochCallback:
    end_training = False
    not_improved_epoch = 0
    monitor_value = np.inf

    def __init__(self, model_name, total_epoch_num, model, optimizer, monitor=None, patience=None, log_file=None):
        if isinstance(model_name, str):
            model_name = [model_name]
            model = [model]
            optimizer = [optimizer]

        self.model_name = model_name
        self.total_epoch_num = total_epoch_num
        self.monitor = monitor
        self.patience = patience
        self.model = model
        self.optimizer = optimizer
        self.best_performance = None
        self.log_file = log_file

    def __save_model(self):
        for m_name, m, opt in zip(self.model_name, self.model, self.optimizer):
            torch.save({'model_state_dict': m.state_dict(),
                        'optimizer_state_dict': opt.state_dict()},
                       m_name)

            print(f'Model saved to {m_name}')

    def epoch_end(self, epoch_num, hash):
        epoch_end_str = f'Epoch {epoch_num}/{self.total_epoch_num} - '
        for name, value in hash.items():
            epoch_end_str += f'{name}: {round(value, 4)} '

        print(epoch_end_str)

        if self.monitor is None:
            self.__save_model()

        elif hash[self.monitor] < self.monitor_value:
            print(f'{self.monitor} decreased from {round(self.monitor_value, 4)} to {round(hash[self.monitor], 4)}')

            self.not_improved_epoch = 0
            self.monitor_value = hash[self.monitor]
            self.best_performance = hash[self.monitor]
            self.__save_model()
        else:
            print(f'{self.monitor} did not decrease from {round(self.monitor_value, 4)}, model did not save!')

            self.not_improved_epoch += 1
            if self.patience is not None and self.not_improved_epoch >= self.patience:
                print("Training was stopped by callback!")
                self.end_training = True
