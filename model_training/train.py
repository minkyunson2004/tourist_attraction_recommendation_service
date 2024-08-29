import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, train_csv, place_name_to_embedding_csv):
        self.train_csv = pd.read_csv(train_csv)
        self.place_name_to_embedding_csv = pd.read_csv(place_name_to_embedding_csv)

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        information_not_include_embedding = list(self.train_csv.iloc[idx, 1:9])
        embedding_information = list(self.place_name_to_embedding_csv[self.train_csv['VISIT_AREA_NM'][idx]])
        information = information_not_include_embedding + embedding_information
        information = torch.FloatTensor(information)
        label = self.train_csv['DGSTFN'][idx]
        label = label.astype(np.float32).reshape(1)

        return information, label


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(1032, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1)
        )

    def forward(self, x):
        return self.sequential(x)


def train(model, train_loader, optimizer, batch_size):
    model.train()
    for batch_idx, (data, answer) in enumerate(train_loader):
        if data.shape[0] != batch_size:
            break

        prediction = model(data)
        loss = nn.MSELoss()(prediction, answer)

        # 학습 루프 내에서
        if torch.isnan(loss):
            print(f"NaN 발생! epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss}")
            print("Input data:", data)
            print("Prediction:", prediction)
            print("Target:", answer)
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch_idx + 1) * batch_size
        if batch_idx % 8 == 0:
            print("lose of current train batch: " + str(loss) + ", (" + str(current) + '/' + str(len(train_loader.dataset)) + ')')


def test(model, test_loader, batch_size):
    model.eval()
    loss = 0

    with torch.no_grad():
        for batch_idx, (data, answer) in enumerate(test_loader):
            prediction = model(data)
            current_loss = nn.MSELoss()(prediction, answer).item()
            if batch_idx % 8 == 0:
                print("current test batch loss: " + str(current_loss))
            loss += current_loss

    print("average test batch loss: " + str(loss / (len(test_loader))) + '\n')
    return loss / (len(test_loader.dataset) // batch_size)


train_dataset = pd.read_csv('data_preprocessing/train.csv')
train_dataset = train_dataset.sample(frac=1)
test_dataset = train_dataset[2048:]
train_dataset = train_dataset[:2048]
train_dataset.to_csv('model_training/training.csv', index=False)
test_dataset.to_csv('model_training/testing.csv', index=False)

train_data = CustomDataset(train_csv='model_training/training.csv', place_name_to_embedding_csv='data_preprocessing/place_name_to_embedding.csv')
test_data = CustomDataset(train_csv='model_training/testing.csv', place_name_to_embedding_csv='data_preprocessing/place_name_to_embedding.csv')

batch_size = 16
epochs = 100
learning_rate = 0.00001

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

regression_model = RegressionModel()
optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate)

continuous_decrease_in_accuracy = 0
min_lose = 10000000

for epoch in range(epochs):
    print("-------- Epoch " + str(epoch + 1) + ' --------')
    train(regression_model, train_dataloader, optimizer, batch_size)
    current_loss = test(regression_model, test_dataloader, batch_size)

    # early stopping
    if current_loss > min_lose:
        continuous_decrease_in_accuracy = continuous_decrease_in_accuracy + 1
    else:
        min_lose = current_loss
        continuous_decrease_in_accuracy = 0
        torch.save(regression_model.state_dict(), 'model_training/model.pth')
    if continuous_decrease_in_accuracy == 10:
        break

print('Finished Training')
