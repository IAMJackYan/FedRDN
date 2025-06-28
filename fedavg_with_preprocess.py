import logging
import torch
import numpy as np
import random
import argparse
import os

from utils import set_for_logger
from pathlib import Path
from nets import AlexNet
import copy


import torchvision.transforms as transforms
import pickle as pkl
from utils import OfficeDataset

def statisc_data(data_loaders):
    means = []
    stds = []
    for data_loader in data_loaders:
        channel_mean = torch.zeros(3)
        channel_std = torch.zeros(3)
        n_samples = 0.
        for data, _ in data_loader:
            N, C, H, W = data.shape
            data = data.view(N, C, -1)
            channel_mean += data.mean(2).sum(0)
            channel_std += data.std(2).sum(0)
            n_samples += N
        channel_mean /= n_samples
        channel_std /= n_samples
        means.append(channel_mean)
        stds.append(channel_std)
    return means, stds


def Normalize(x, means, stds):
    mean = random.choice(means)
    std = random.choice(stds)        
    for i in range(3):
        x[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return x

def test(model, data_loader, loss_fun, device, preprocess=False, means=None, stds=None):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            if preprocess:
                data = Normalize(data, means, stds)

            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total

def train_fedavg(net_id, net, train_dataloader, epochs, lr, optimizer, weight_decay, device, preprocess=False, means=None, stds=None):
    logging.info('client training %s' % str(net_id))
    net.train()
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay,
                               amsgrad=True)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(1, epochs+1):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            
            if preprocess:
                x = Normalize(x, means, stds)

            x, target = x.to(device), target.to(device)
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / (len(epoch_loss_collector)+ 1e-14)
        logging.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    logging.info(' ** Training complete **')
    return epoch_loss


def prepare_data(args):
    data_base_path = './data/'
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),          
            transforms.ToTensor(),
    ])
    
    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True, num_workers=4)
    amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=args.batch, shuffle=False, num_workers=4)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False, num_workers=4)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True, num_workers=4)
    caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=args.batch, shuffle=False, num_workers=4)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False, num_workers=4)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True, num_workers=4)
    dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=args.batch, shuffle=False, num_workers=4)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False, num_workers=4)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True, num_workers=4)
    webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=args.batch, shuffle=False, num_workers=4)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False, num_workers=4)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]


    amazon_weight = len(amazon_trainset) / (len(amazon_trainset) + len(caltech_trainset) + len(dslr_trainset) + len(webcam_trainset))
    caltech_weight = len(caltech_trainset)/ (len(amazon_trainset) + len(caltech_trainset) + len(dslr_trainset) + len(webcam_trainset))
    dslr_weight = len(dslr_trainset) / (len(amazon_trainset) + len(caltech_trainset) + len(dslr_trainset) + len(webcam_trainset))
    webcam_weight = len(webcam_trainset) / (len(amazon_trainset) + len(caltech_trainset) + len(dslr_trainset) + len(webcam_trainset))

    return train_loaders, val_loaders, test_loaders, [amazon_weight, caltech_weight, dslr_weight, webcam_weight]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:3', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--batch', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--fl_method', type=str, default='fedavg')
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--kd_mode', type=str, default='cse')
    parser.add_argument('--preprocess_way', type=str, default='base')
    parser.add_argument('--preprocess', action='store_true', default=False)

    args = parser.parse_args()
    return args

def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    train_loaders, val_loaders, test_loaders, fed_avg_freqs = prepare_data(args)
    global_model = AlexNet().to(device)
    
    datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    client_num = 4

    best_accuarcy = 0
    best_round = 0

    local_models = [copy.deepcopy(global_model).to(device) for idx in range(client_num)]

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    weight_save_dir = os.path.join(args.save_dir, args.partition, args.fl_method, str(os.getpid()))
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    if args.preprocess_way == 'base':
        means = [[0.485, 0.456, 0.406]]
        stds = [[0.229, 0.224, 0.225]]
    elif args.preprocess_way == 'ours':
        means, stds = statisc_data(train_loaders)

    for round in range(1, args.comm_round):

        logging.info('----Communication Round: %d -----' % round)
        global_w = global_model.state_dict()

        for i in range(client_num):
            
            train_fedavg(i, local_models[i], train_loaders[i], args.epochs, args.lr, args.optimizer, args.weight_decay, device, args.preprocess, means, stds)

        for id in range(client_num):
            model_param = local_models[id].state_dict()
            for key in model_param:
                if id == 0:
                    global_w[key] = model_param[key] * fed_avg_freqs[id]
                else:
                    global_w[key] += model_param[key] * fed_avg_freqs[id]

        global_model.load_state_dict(global_w)
        for i in range(client_num):
            local_models[i].load_state_dict(global_w)

        avg_acc = 0

        logging.info('------------------Testing-----------------')
        for i in range(client_num):
            if args.preprocess_way == 'ours':
                _, test_acc = test(global_model, test_loaders[i], loss_fn, device, args.preprocess, [means[i]], [stds[i]])
            else:
                _, test_acc = test(global_model, test_loaders[i], loss_fn, device, args.preprocess, means, stds)
            logging.info('>> %s Test accuracy: %f' % (datasets[i], test_acc))
            avg_acc += test_acc
        
        avg_acc /= client_num
        logging.info('>> Average Test accuracy: %f' %avg_acc)

        if avg_acc > best_accuarcy:
            best_accuarcy = avg_acc
            best_round = round


    logging.info(' %d epoch get the best acc %f' % (best_round, best_accuarcy))

if __name__ == '__main__':
    args = get_args()
    main(args)