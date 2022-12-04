from typing import List, Tuple
from PIL import Image
from os import listdir
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

def dir_to_list(directory: str) -> List[str]:
    ''' Returns a list of the full paths of files in the directory.

        Args:
        - directory: (str) path to the directory containing the files. 

        Returns:
        - list of paths as str.

    '''
    return [join(directory, file) for file in listdir(directory)]


def ds_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std



class DeepfakeDataset(Dataset):
    '''
        Dataset class for the real and fake faces
    '''
    def __init__(
            self,
            img_dir: str,
            version: str,
            transform=None,
        ) -> None:
        '''
        Args:
        - img_dir: folder where all images are located
        - version: "train", "validation" or "test"
        - transform: transformations to perform to the image
        '''
        self.img_dir = img_dir
        self.version = version 
        self.transform = transform
        self.real_path = join(img_dir, f'{version}/real')
        self.fake_path = join(img_dir, f'{version}/fake')

        self.images = dir_to_list(self.real_path) + dir_to_list(self.fake_path) 

        self.tot_img = len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = f'{self.images[idx]}'
        img = read_image(img_path)/255.
        
        # get label
        label = 1 if "real" in img_path else 0

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim,
    opt_scheduler: torch.optim.lr_scheduler = None,
    device: str = "cpu",
) -> torch.FloatTensor:
    """
    Backpropagation step to train the model.

    Args:
    - dataloader:
    - model:
    - loss_fn:
    - optimizer:
    - opt_scheduler:
    - device:

    Returns: current mean loss value
    """
    model.train()

    losses = []
    for x_train, y_train in dataloader:
        x_train, y_train = x_train.to(device), y_train.to(device)
        # Compute prediction error
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    
    if opt_scheduler is not None:
        opt_scheduler.step()

    return torch.FloatTensor(losses).mean()


def eval(
    dataloader: DataLoader,
    model: nn.Module, 
    loss_fn: nn.Module,
    device: str = "cpu",
) -> Tuple[torch.FloatTensor, int]:
    """
    Evaluates a model on a dataloader object.

    Args:
    - dataloader: dataloader pytorch object defining the dataset
    - model: trained model to evaluate
    - loss_fn: loss function to calculate loss
    - device: device where to run the evaluation

    Returns: the mean loss and precision of the model on the given dataset
    """
    model.eval()

    losses = []
    tot_size = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for x_eval, y_eval in dataloader:
            x_eval, y_eval = x_eval.to(device), y_eval.to(device)
            # make prediction
            pred = model(x_eval)
            # calculate loss
            loss = loss_fn(pred, y_eval)
            losses.append(loss)
            # calculate correctly classified images
            correct += (pred.argmax(1) == y_eval).type(torch.float).sum().item()
    correct /= tot_size
    losses = torch.FloatTensor(losses)
    return losses.mean(), correct


def predict(
    dataloader: DataLoader,
    model: nn.Module, 
    device: str,
) -> Tuple[np.array, np.array]:
    """
    Predicts the label for the given dataset.

    Args:
    - dataloader: dataset for which to predict the labels
    - model: model to use for the prediction
    - device: device on which to perform calculations

    Returns: np.array of predictions and true labels.
    """
    model.eval()

    preds = []
    all_y = []
    with torch.no_grad():
        for x_data, y_data in dataloader:
            all_y.append(y_data)
            x_data, y_data = x_data.to(device), y_data.to(device)
            # make prediction
            pred = model(x_data).cpu().numpy()
            preds.append(pred)
    preds = np.argmax(np.concatenate(preds, axis=0), axis=1)
    all_y = np.concatenate(all_y, axis=0)
    return preds, all_y

def save_train_plot(
    train_loss: List[float],
    eval_loss: List[float],
    plot_dir: str,
    plot_name: str,
) -> None:
    """
    Saves the training plots.

    Args:
    - train_loss and eval_loss: are the lists of the respective losses
    - plot_dir: directory where to save the plots
    - plot_name: name of the file and title of the plot
    """
    plt.plot(train_loss, label="train")
    plt.plot(eval_loss, label="evaluate")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title(plot_name)
    plt.savefig(join(plot_dir, plot_name+".png"))
