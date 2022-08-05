import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

##parameter
num_epochs = 5000
learning_rate = 0.001

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


class Dataset:
    def __init__(self, filename):
        self.x = None
        self.y = None

        data = pd.read_csv(filename, sep=",")

        self.x = data['x']
        self.y = data['y']

    def getDataset(self):
        return self.x, self.y


class simple_MLP(nn.Module):
    def __init__(self):
        super(simple_MLP, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(1, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x

def plot(loss):
    plt.plot(loss)
    plt.title('loss')
    plt.xlabel('epoch')
    plt.show()

def formula_visualization(x_list, y_list, predict_y_list):
    plt.scatter(x_list, y_list, label='GT', c='blue')
    plt.scatter(x_list, predict_y_list, label='predict', c='red')
    plt.legend()
    plt.show()
    pass

def main(filename):
    data_x, data_y = Dataset(filename).getDataset()
    final_data_x, final_data_y = torch.FloatTensor(data_x.to_numpy()), torch.FloatTensor(data_y.to_numpy())

    final_data_x = final_data_x.view(-1, 1)
    final_data_y = final_data_y.view(-1, 1)

    '''load_model'''
    model = simple_MLP()

    '''set loss and optimizer'''
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-7)


    total_loss = []
    for epoch in range(num_epochs):

        optimizer.zero_grad()

        predict_y = model(final_data_x)
        loss = criterion(predict_y, final_data_y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch: {epoch} | MSE loss: {loss}")

        total_loss.append(loss.item())

    # plot(total_loss)

    model.eval()
    with torch.no_grad():
        test_predict_y = model(final_data_x)
        formula_visualization(final_data_x, final_data_y, test_predict_y)



if __name__ == '__main__':
    filename = './data(gp)/data-gp1.txt'
    main(filename)
