import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_model(train, net, epochs=1, device=None, lr=1e-3):
    optimizer = optim.AdamW(net.parameters(), lr)
    loss_history = []
    model = net.to(device)

    model.train()

    for epoch in range(epochs):
        for i, (data, label) in enumerate(train):
            x, y = data.to(device), label.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.CrossEntropyLoss(y_pred,y.long())
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
            
    return loss_history, y_pred


def test_model(test, net, device=None, lr=1e-3):
    optimizer = optim.AdamW(net.parameters(), lr)

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (data, labels) in enumerate(test):
            data = data.to(device)
            labels = labels.to(device).to(torch.float32)

            outputs = net(data).reshape(-1)
            predicted = (outputs > 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    print(f'Test Accuracy: {accuracy:.2%}')


"""
    def train_model(self, train_set, epochs=1, device=torch.device('cpu')):
        loss_list = []
        y_pred = []
        model = self.model.to(device=device)  # move the model parameters to CPU/GPU
        for e in range(epochs):
            for t, (x, y) in enumerate(train_set):
                model.train()  # put model to training mode
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)

                scores = model(x)
                loss = nn.functional.cross_entropy(scores, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())

                # if t % 100 == 0:  # возможно лучше запоминать данные в список и возвращать его?
                # print('Iteration %d, loss = %.4f' % (t, loss.item()))
                # f1_score(loader_val, model)
                # print()
    """
