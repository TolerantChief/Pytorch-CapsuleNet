import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from capsnet import CapsNet
from data_loader import Dataset
from tqdm import tqdm
from torchbearer.callbacks import EarlyStopping
from numpy import prod
from mem_profile import get_gpu_memory_map
import matplotlib.pyplot as plt

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
N_EPOCHS = 500
LEARNING_RATE = 0.001
MOMENTUM = 0.9

'''
Config class to determine the parameters for capsule net
'''


class Config:
    def __init__(self, dataset='mnist'):
        if dataset == 'mnist':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'cifar10':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'Jamones':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 26
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32



def train(model, optimizer, train_loader, epoch):
    global current_memory_usage
    global max_memory_usage
    global history

    capsule_net = model
    capsule_net.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):

        target = torch.sparse.torch.eye(26).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()
        correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        total_accuracy += correct
        train_loss = loss.item()
        total_loss += train_loss
        if batch_id % 100 == 0:
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                correct / float(BATCH_SIZE),
                train_loss / float(BATCH_SIZE)
                ))
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch,N_EPOCHS,total_loss / len(train_loader.dataset)))

    history['train_loss'].append(total_loss / len(train_loader.dataset))
    history['train_acc'].append(total_accuracy / len(train_loader.dataset))

    current_memory_usage = get_gpu_memory_map()[0]
    if current_memory_usage > max_memory_usage:
        max_memory_usage = current_memory_usage



def test(capsule_net, test_loader, epoch):
    global early_stopping
    global best_test_acc
    global best_epoch
    global epochs_no_improve
    global current_memory_usage
    global max_memory_usage
    global dataset
    
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):

        target = torch.sparse.torch.eye(26).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.item()
        correct += sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                       np.argmax(target.data.cpu().numpy(), 1))

    tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(epoch, N_EPOCHS, correct / len(test_loader.dataset),
                                                                  test_loss / len(test_loader)))

    history['test_acc'].append(correct / len(test_loader.dataset))
    history['test_loss'].append(test_loss / len(test_loader))

    current_memory_usage = get_gpu_memory_map()[0]
    if current_memory_usage > max_memory_usage:
        max_memory_usage = current_memory_usage

    accuracy = correct / len(test_loader.dataset)

    if accuracy > best_test_acc:
      best_test_acc = accuracy
      best_epoch = epoch
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1
      if epochs_no_improve == early_stopping.patience:
        print('Early Stopping')
        print(f'The best test accuracy was {best_test_acc} in epoch {best_epoch}')

        print('max. memory usage: ', max_memory_usage)

        epochs = range(1, epoch + 1)
        plt.figure()

        plt.subplot(1, 2, 1)  # 1 row, 2 column, subplot 1 for accuracy
        # Plot accuracy
        plt.plot(epochs, history['train_acc'], label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'CapsNet: {dataset}')
        plt.legend()

        plt.subplot(1, 2, 2)  # 2 rows, 1 column, subplot 2 for loss
        # Plot loss
        plt.plot(epochs, history['train_loss'], label='Training Loss')
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'CapsNet: {dataset}')
        plt.legend()

        plt.tight_layout()  # To ensure subplots are properly arranged
        plt.show()
        return

if __name__ == '__main__':
    #torch.manual_seed(1)
    dataset = 'Jamones'
    # dataset = 'mnist'
    config = Config(dataset)
    mnist = Dataset(dataset, BATCH_SIZE)

    capsule_net = CapsNet(config)
    capsule_net = torch.nn.DataParallel(capsule_net)
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    capsule_net = capsule_net.module

    optimizer = torch.optim.Adam(capsule_net.parameters())

    early_stopping = EarlyStopping(patience=10)
    best_test_acc = 0
    best_epoch = 0
    epochs_no_improve = 0

    current_memory_usage = get_gpu_memory_map()[0]
    max_memory_usage = 0

    print('Num params:', sum([prod(p.size())
              for p in capsule_net.parameters()]))

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, mnist.train_loader, e)
        test(capsule_net, mnist.test_loader, e)

        if epochs_no_improve == early_stopping.patience:
          break
