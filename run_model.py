import torch
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import torch
from torch import nn, optim, utils

"""preprocess = T.Compose([
   T.Resize(256),
   T.CenterCrop(224),
   T.ToTensor(),
   T.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )
])"""

class CNN(nn.Module):

    def __init__(self):
        """Initialize the CNN network and define its layers"""
        super().__init__()

        # there are many different ways of defining a PyTorch model, nn.Sequential is just the most
        # convenient one as everything is in one callable 'thing' and executes in sequence
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 2))
    
    def forward(self, x):
        """Pass input x through the network to obtain an output"""
        output = self.network(x)
        return output

    def size(self):
        """Count the number of parameters (i.e., size of weights) in the network"""
        parameters = self.parameters()
        size = 0
        for parameter in parameters:
            size += np.prod(parameter.shape)
        return size

transform = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float)])

def read_and_process_img(file_name):
    im = Image.open(f'static/imgs/{file_name}')
    pp = transform(im)
    return pp.unsqueeze(0)

def update_predictions():
    imgs = pd.read_csv('static/img_data.csv', header=0)
    imgs_unlabeled = imgs[imgs['Class'] == 'U']
    inputs = torch.vstack(tuple(imgs_unlabeled['File_Name'].apply(read_and_process_img)))

    model = CNN()
    model.load_state_dict(torch.load('static/cifar10_model_binary.pth', map_location=torch.device('cpu')))
    model.eval()
    outputs = model(inputs)
    predictions = outputs.argmax(dim=-1)
    # with open("test_output", "w+") as f:
    #     f.write(predictions)
    imgs.loc[imgs['Class'] == 'U', 'Class'] = predictions
    imgs.to_csv('static/img_data.csv', index=False)