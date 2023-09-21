import torch
import torch.optim as optim
import torch.nn as nn


def train_model(train, net, epochs=1, device=None, lr=1e-3):
    optimizer = optim.AdamW(net.parameters(), lr)
    loss_history = []
    model = net.to(device)

    model.train()

    for epoch in range(epochs):
        for i, (data, label) in enumerate(train):
            x, y = data.to(device), label.to(device)
            optimizer.zero_grad()
            y_pred = model(x) # ошибка тут
            loss = nn.CrossEntropyLoss(y_pred,y.long())
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
            
    return loss_history, y_pred


def test_model(test, net, device=None, lr=1e-3):
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

