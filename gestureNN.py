
import torch
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd

datapath = ("data\mock_gesture_data.csv")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
learning_rate = 1e-3
batch_size = 64
num_gestures =0

class HandsLandmarkDataset(Dataset):
    def __init__(self,datapth):
        df = pd.read_csv(datapth, header=None)
        self.classes = list(set(df[0].values))          # unique gesture names
        self.labels = [self.classes.index(l) for l in df[0].values]  # names → numbers
        self.features = df.iloc[:, 1:].values     #all samples, each row is a captures frame for wcam


    def normalise(self,landmarks):
        landmarks= landmarks.reshape(21,3)
        wrist = landmarks[0]
        landmarks=landmarks-wrist
        return landmarks.flatten()

    def __getitem__(self, idx):
        features =self.normalise(self.features[idx])
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.labels)
    

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(63, 128),
        nn.ReLU(),
         nn.Dropout(0.3),      # randomly drop 30% of neurons during training
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_gestures))
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


dataset = HandsLandmarkDataset(datapath)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = NeuralNet().to(device)



def train_loop(dataloader, model, loss_fn,optimiser):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x,y) in enumerate(dataloader):  # where X is the image tensor, and y is the ACTUAL Class


        predic = model(x)
        loss = loss_fn(predic,y) # calculates loss of NN, using the prediction and the real valyes

        # backpropogation algorithim - are we isacc chat?
        loss.backward() # computes the gradients of loss backpropogation
        optimiser.step() # update the NN Parametres using the gradients
        optimiser.zero_grad() # zero the gradients

        if batch %100==0:
            loss,current_data = loss.item(), batch* batch_size + len(x)
            img = dataloader.dataset[current_data][0]
            #        plt.imshow(img.permute(1,2,0))
            #        plt.axis('off')
            #        plt.show()
            print(f"loss: {loss:>7f}  [{current_data:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_fn):
    model.eval() # evaluation mode, no dropping neurons
    size = len(dataloader.dataset)
    num_batches= len(dataloader.dataset)
    test_loss, correct = 0,0
    with torch.no_grad():#no learning or backpropogation so no gradients needed
        for x,y in dataloader:
            pred=model(x)
            test_loss += loss_fn(pred,y).item() # calculate total    loss/cost
            correct+= (pred.argmax(1)== y).type(torch.float).sum().item()# smart way of counting the amount of correct predidctions
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



loss_fn = nn.CrossEntropyLoss() # loss functin is the mean of the square differences
optimiser =torch.optim.Adam(model.parameters(),lr =learning_rate) #use stoic GD to minimize loss
epochs =100 #loop through data set 10 times
for test in range(epochs):
    print(f"Epoch {test + 1}\n-------------------------------")
    train_loop(loader, model, loss_fn, optimiser)
    test_loop(loader, model, loss_fn)