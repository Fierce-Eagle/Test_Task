import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_model(loader_train, net, epochs=10, device=None, lr=1e-3):
    assert device is not None, "device must be cpu or cuda"
    сrossEntropy = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr)
    loss_history = []
    acc_history = []
    y_pred_history = []
    model = net.to(device)

    #model.train()

    for epoch in range(epochs):
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.int64)
            y = torch.flatten(y)
            scores = model(x)
            loss = сrossEntropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_history.append(scores)
            
            if t % 10 == 0:
                pred = torch.argmax(scores, dim=1)
                correct = pred.eq(y)
                acc = torch.mean(correct.float())
                
                print('Iteration %d, loss = %.4f acc = %.4f' % (t, loss.item(), acc))
                
                loss_history.append(loss)
                acc_history.append(acc)
            
        print()

    return loss_history, acc_history, y_pred_history


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
