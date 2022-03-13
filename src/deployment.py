from src.model import BaseMLP
from src.custom_dataset import CustomDataset
from src import FocalLoss
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# training baseMLP with simple hold-out CV
def train_mlp(train_df: pd.DataFrame, val_df: pd.DataFrame, batch_size=10, num_iter=100, gamma=0, alpha=None, gpu=False):
    """
    :param train_df: dataframe of training data
    :param val_df: dataframe of validation data
    :param batch_size: the batch size for dataloader, defaults to 10
    :param num_iter: number of epochs, defualts to 100
    :param gamma: the gamma parameter for FocalLoss Function, defaults to 0
    :param gpu: specify to use GPU or CPU
    :return:
    """

    net = BaseMLP(21)
    loss_function = FocalLoss.FocalLoss(gamma=gamma)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    if gpu:
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        print(f'running on device: {torch.cuda.get_device_name(device)}')
        # move neural network modules onto gpu
        net.to(device)
        loss_function.to(device)

    torch_train = CustomDataset(train_df)
    torch_val = CustomDataset(val_df)
    train_loader = DataLoader(torch_train, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(torch_val, batch_size=batch_size, shuffle=True, num_workers=1)

    avg_train_loss = []
    avg_val_loss = []
    lowest_val_loss = float('inf')
    early_stop = 0
    # for every epoch
    for epoch in range(num_iter):
        # print(f'---------------at epoch: {epoch + 1}---------------')
        batch_train_losses = []
        batch_val_losses = []

        # set the mode of the model to train and start batch training
        net.train()
        # for every batch in the data from train loader
        for i, batch_data in enumerate(train_loader):
            # get batch X and batch y
            inputs, labels = batch_data
            if gpu:
                # move tensors onto gpu
                inputs, labels = inputs.to(device), labels.to(device)

            # zero out the gradients
            optimizer.zero_grad()
            # perform prediction
            outputs = net(inputs)
            # check loss
            loss = loss_function(outputs, labels)
            # back propagates gradient
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        # get average train loss in this epoch
        one_train_loss = np.mean(batch_train_losses)
        avg_train_loss.append(one_train_loss)

        # stop training the model, set to eval() mode for validation set
        net.eval()
        for inputs, labels in val_loader:

            if gpu:
                # move tensors onto gpu
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            batch_val_losses.append(loss.item())
        # print(f'output layer looks like this: {outputs}')
        one_val_loss = np.mean(batch_val_losses)
        avg_val_loss.append(one_val_loss)
        # if validation performance of this epoch is better than that of the previous one
        if one_val_loss < lowest_val_loss:
            lowest_val_loss = one_val_loss
            early_stop = epoch
            torch.save(net.state_dict(), f"model/epoch{early_stop}.pt")
            i += 1
        # print model performance every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}: Train loss: {one_train_loss}, Validation loss: {one_val_loss}")
    print('training finished')
    net.load_state_dict(torch.load(f'model/epoch{early_stop}.pt'))
    return net, avg_train_loss, avg_val_loss


def mlp_predict(net, data, use_gpu=True):
    """
    this function is only used for non-batched prediction tasks.
    :param net: the model
    :param data: dataframe
    :param use_gpu: whether or not to move tensors onto gpu
    :return:
    """
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    # make sure not doing grad descent
    dataset = CustomDataset(data)
    # data is small enough, therefore do it in memory
    inputs = dataset.x_train
    labels = dataset.y_train
    logit = nn.Softmax(dim=1)

    if use_gpu:
        net.to(device)
        inputs.to(device)
        labels.to(device)
        logit.to(device)
    net.eval()
    ps = net(inputs)
    return torch.argmax(logit(ps), dim=1), dataset.y_train


# The pseudo run script that acts as a pipeline from training the model to presenting performance
def mlp_run(train_df: pd.DataFrame, val_df: pd.DataFrame, batch_size=10, num_iter=100,
            gamma=0, alpha=None, gpu=False):
    """
    :param train_df:
    :param val_df:
    :param batch_size:
    :param num_iter:
    :param gamma:
    :param alpha:
    :param gpu:
    :return:

    current bug: gpu=True doesn't work due to more than 1 device error, even though the two functions work
    separately when gpu=True is passed in
    """
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    net, train_loss, val_loss = train_mlp(train_df, val_df, batch_size=batch_size, num_iter=num_iter, gamma=gamma,
                                          alpha=alpha,
                                          gpu=gpu)
    x_axis = list(range(1, num_iter + 1))
    plt.plot(x_axis, train_loss, label='train_loss')
    plt.plot(x_axis, val_loss, label='val_loss')
    plt.legend()
    plt.show()
    # print('did plot show?')
    if gpu:
        net.to(device)
    y_train_pred, y_train = mlp_predict(net, train_df, use_gpu=gpu)
    y_val_pred, y_val = mlp_predict(net, val_df, use_gpu=gpu)
    print(f'accuracy = {torch.mean((y_train_pred == y_train).type(torch.float))}')
    print(f'accuracy = {torch.mean((y_val_pred == y_val).type(torch.float))}')
    matrix_train = ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred))
    matrix_val = ConfusionMatrixDisplay(confusion_matrix(y_val, y_val_pred))
    matrix_train.plot()
    plt.show()
    matrix_val.plot()
    plt.show()
    return net





