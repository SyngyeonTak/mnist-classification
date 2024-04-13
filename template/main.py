import os
import dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from model import LeNet5, CustomMLP
import matplotlib.pyplot as plt

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer, epoch, num_epochs):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here

    model.train()
    trn_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trn_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # it moves inputs and targets target to the specified device (GPU or CPU)

        optimizer.zero_grad()
        #  before computing the gradients for a new batch of data
        #, it's important to reset (zero out) the gradients; 
        # otherwise, gradients would accumulate across batches, leading to incorrect updates.

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        # This line computes the loss by passing the model outputs and the target values to the specified loss function
        # loss value's type would typically be a scalar tensor. 

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # This line updates the parameters of the model using the computed gradients. 
        # It adjusts the parameters in the direction that minimizes the loss, typically 
        # using some variant of stochastic gradient descent (SGD) or its extensions (e.g., Adam, RMSprop). 

        # Accumulate loss
        trn_loss += loss.item() * inputs.size(0)
        # loss.item() is the average loss
        # if you want to learn it with the total batch loss you can simply multiply it by the batch size that is inputs.size(0)

         # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
            # this will find the maximum value in the specified dimension which 1 in this case
            # The first element is a tensor containing the maximum values
            # The second element is a tensor containing the indices of the maximum values (we are interested in this)

        if batch_idx % 50 == 0:  # Print every 500 mini-batches
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(trn_loader)}], Loss: {loss.item():.4f}")

        correct += (predicted == targets).sum().item()
        # The item() method in PyTorch is used to extract a single scalar value from a tensor containing a single element.
        total += targets.size(0)

        trn_loss /= len(trn_loader.dataset)
        # this is for calculating average loss

        acc = correct / total

        #print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {trn_loss:.4f}, Train Acc: {acc:.4f}")


    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    tst_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # The with torch.no_grad() context manager is commonly used in the test/validation phase of a machine learning model for several reasons:
        # During inference or evaluation, you typically don't need to compute gradients because you're not updating the model parameters

        for inputs, targets in tst_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward Pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Accmulate loss
            tst_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            acc = correct/total

            tst_loss /= len(tst_loader.dataset)

    return tst_loss, acc

def train_and_test(model, applied_train_loader, applied_test_loader, device, criterion, optimizer, num_epochs, model_name):
    model.to(device)
    
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, applied_train_loader, device, criterion, optimizer, epoch, num_epochs)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        test_loss, test_acc = test(model, applied_test_loader, device, criterion)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

    
    print(f"model Name: {model_name} \n Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return train_loss_history, train_acc_history, test_loss_history, test_acc_history

def plot_performance(train_loss_history, train_acc_history, test_loss_history, test_acc_history, model_name):
    epochs = range(1, len(train_loss_history) + 1)

    print('epochs: ', epochs)
    print('train_loss_history: ', train_loss_history)
    print('test_loss_history: ', test_loss_history)



    # Plot training and testing loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_history, label='Training Loss')
    plt.plot(epochs, test_loss_history, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'../img/{model_name}_loss_plot.png')

    # Plot training and testing accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc_history, label='Training Accuracy')
    plt.plot(epochs, test_acc_history, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'../img/{model_name}_accuracy_plot.png')

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    # 

    # Instantiate Dataset objects for training and test datasets
    #train_dataset = datasets.MNIST(root='../data/train.tar', train=True, transform=transform, download=True)
    #test_dataset = datasets.MNIST(root='../data/test.tar', train=False, transform=transform, download=True)

    current_directory = os.getcwd()
    print("Current directory:", current_directory)

    train_dataset = dataset.MNIST(data_dir='../data/train.tar', apply_augumentation = False)
    test_dataset = dataset.MNIST(data_dir='../data/test.tar', apply_augumentation = False)
    

    train_aug_dataset = dataset.MNIST(data_dir='../data/train.tar', apply_augumentation = True)
    #test_aug_dataset = dataset.MNIST(data_dir='../data/test.tar', apply_augumentation = True)

    
    # Instantiate DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    train_aug_loader = DataLoader(train_aug_dataset, batch_size=500, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD
    criterion = nn.CrossEntropyLoss()
    num_epochs = 25

    # Train and test with LeNet5
    model_configs = [
        {"model": LeNet5(dropout=False), "optimizer_params": {"lr": 0.01, "momentum": 0.9}, "model_name": "LeNet"},
        {"model": LeNet5(dropout=True), "optimizer_params": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.01}, "model_name": "LeNet_regularization"}, #  regularization 2  - Weight Decay and Dropout
        {"model": CustomMLP(), "optimizer_params": {"lr": 0.01, "momentum": 0.9}, "model_name": "CustomMLP"}
    ]


    # Train and test each model configuration
    for config in model_configs:
        model = config["model"]
        optimizer_params = config["optimizer_params"]
        model_name = config["model_name"]
        optimizer_instance = optimizer([{"params": model.parameters(), **optimizer_params}])
        
        if model_name == 'LeNet_regularization':    
            applied_train_loader = train_aug_loader
        else:
            applied_train_loader = train_loader
            
        applied_test_loader = test_loader    
        train_loss_history, train_acc_history,test_loss_history, test_acc_history = train_and_test(model, applied_train_loader, applied_test_loader, device, criterion, optimizer_instance, num_epochs, model_name)
        
        plot_performance(train_loss_history, train_acc_history, test_loss_history, test_acc_history, model_name)

if __name__ == '__main__':
    main()
