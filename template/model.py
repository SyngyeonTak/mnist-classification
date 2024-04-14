import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self, dropout):
        # write your codes here
        super(LeNet5, self).__init__() # this will call the super class of LeNet5 class, self arguement is the current instance

        self.features = nn.Sequential( 
            # self.features refers to a sequential module that defines the feature extraction part of the LeNet-5 architecture
            # Sequential() allows to construct a neural network by arranging and encapsulating multiple layers or modules in a sequential order.
            nn.Conv2d(in_channels=1 , out_channels=6, kernel_size=5), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(in_channels=6 , out_channels=16, kernel_size=5), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

        )
    
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Dropout(p = 0.5) if dropout else nn.Identity(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Dropout(p= 0.5) if dropout else nn.Identity(),
            nn.Linear(in_features=84, out_features=10) 
        )
        


    def forward(self, img):
        features = self.features(img)
        output = self.classifier(features)

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        # write your codes here
        super(CustomMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=77),
            nn.ReLU(),
            nn.Linear(in_features=77, out_features=10) 
        )

    def forward(self, img):

        # write your codes here
        output = self.classifier(img)
        return output
