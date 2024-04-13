import os
import torch
import tarfile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import some packages you need here

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, apply_augumentation = False):
        # write your codes here
        self.data_dir = data_dir

        # Extract labels, file paths, and load images into memory
        self.labels = []
        self.images = []
        
        if apply_augumentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),     # regularization 1  - data augumentation
                #transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                transforms.Resize(32),
                transforms.ToTensor(),                   
                transforms.Normalize((0.1307,), (0.3081,)) 
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),                   
                transforms.Normalize((0.1307,), (0.3081,))  
            ])
        
        
        with tarfile.open(self.data_dir, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    filename = os.path.basename(member.name)
                    label = int(filename.split('_')[1].split('.')[0])
                    self.labels.append(label)
                    image = Image.open(tar.extractfile(member)).convert('L')
                    self.images.append(image)
    
    

    def __len__(self):

        # write your codes here
        return len(self.labels)

    def __getitem__(self, idx):
        # write your codes here
        image = self.images[idx]
        label = self.labels[idx]

        
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == '__main__':

    # write test codes to verify your implementations
    
    train_dataset = MNIST(data_dir='../data/train.tar')
    test_dataset = MNIST(data_dir='../data/test.tar')


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Test printing out some samples

    current_directory = os.getcwd()
    print("Current directory:", current_directory)

    # Access elements from train_loader
    for batch_idx, (images, labels) in enumerate(train_loader):
        # `images` is a batch of images
        # `labels` is a batch of corresponding labels
        # Access elements within the batch here
        print("Batch:", batch_idx)
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        
        # Access individual images and labels within the batch
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            # Do something with the image and label
        
        # Break loop after one iteration (remove this line if you want to loop through all batches)
        break

