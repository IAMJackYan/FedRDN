import time
import os
import logging
import torch
import numpy as np

from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def set_for_logger(args):

    log_filename = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '.txt'
    log_filepath = os.path.join(args.log_dir, log_filename)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)



def compute_accuracy(model, dataloader, get_confusion_matrix=False, device=None, multiloader=False):
    model.eval()
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss().to(device)

    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    x, target = x.to(device), target.to(device)
                    target = target.long()

                    out = model(x)
                    loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x, target = x.to(device), target.to(device)
                target = target.long()
                out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss


# class WramupMu(object):
# #     def __init__(self, init_mu, last_mu, warmup_step, last_warmup_round,):
# #         super(WramupMu, self).__init__()
# #         self.mu = init_mu
# #         self.last_warmup_round = last_warmup_round
# #         self.warmup_factor = (last_mu - init_mu) * warmup_step / last_warmup_round
# #         self.warmup_step = warmup_step
# #
# #     def step(self, round):
# #         if round <= self.last_warmup_round and round % self.warmup_step == 0:
# #             self.mu = self.mu + self.warmup_factor
# #
# # def set_requires_grad(net, requires_grad=False):
# #     """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
# #     Parameters:
# #         nets (network list)   -- a list of networks
# #         requires_grad (bool)  -- whether the networks require gradients or not
# #     """
# #     for param in net.parameters():
# #         param.requires_grad = requires_grad

class WramupMu(object):
    def __init__(self, init_mu, last_mu, warmup_step, last_warmup_round,):
        super(WramupMu, self).__init__()
        self.mu = init_mu
        self.last_warmup_round = last_warmup_round
        self.warmup_factor = (last_mu - init_mu) * warmup_step / last_warmup_round
        self.warmup_step = warmup_step

    def step(self, round):
        if round <= self.last_warmup_round and round % self.warmup_step == 0:
            self.mu = self.mu + self.warmup_factor

class StepMu(object):
    def __init__(self, init_mu, step, factor=10):
        super(StepMu, self).__init__()
        self.mu = init_mu
        self.warmup_step = step
        self.factor = factor
    def step(self, round):
        print(round)
        if round % self.warmup_step == 0:
            self.mu = self.mu * self.factor

def set_requires_grad(net, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    for param in net.parameters():
        param.requires_grad = requires_grad



class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)
            print(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
