import torch
from torchvision import datasets, transforms
import torchvision
import os


class Dataset:
    def __init__(self, dataset, _batch_size):
        super(Dataset, self).__init__()
        if dataset == 'mnist':
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_dataset = datasets.MNIST('/data/mnist', train=True, download=True,
                                           transform=dataset_transform)
            test_dataset = datasets.MNIST('/data/mnist', train=False, download=True,
                                          transform=dataset_transform)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

        elif dataset == 'cifar10':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(
                '/data/cifar', train=True, download=True, transform=data_transform)
            test_dataset = datasets.CIFAR10(
                '/data/cifar', train=False, download=True, transform=data_transform)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False)
        elif dataset == 'Jamones':
            data_transform = transforms.Compose([
                transforms.Resize((self.img_size,self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.RandomRotation(10),
                transforms.RandomCrop(self.img_size, padding=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            test_size = 0.2
            dataset = torchvision.datasets.ImageFolder(root='/content/data/JAMONES', transform=data_transform)
            num_data = len(dataset)
            num_test = int(test_size * num_data)
            num_train = num_data - num_test
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
            train_dataset = datasets.CIFAR10(
                '/data/cifar', train=True, download=True, transform=data_transform)
            test_dataset = datasets.CIFAR10(
                '/data/cifar', train=False, download=True, transform=data_transform)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


        elif dataset == 'office-caltech':
            pass
        elif dataset == 'office31':
            pass
