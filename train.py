import torch
import torch.optim as optim
import torch.nn as nn


def train_model(train, net, epochs=10, device=None, lr=1e-3):
    assert device is not None, "device must be cpu or cuda"
    optimizer = optim.AdamW(net.parameters(), lr)
    loss_history = []
    y_predict_history = []
    model = net.to(device)

    model.train()

    for epoch in range(epochs):
        for image, label in train:
            x = image.to(device=device, dtype=torch.float32)
            y = label.to(device=device,  dtype=torch.long)
            optimizer.zero_grad()
            y_predict = model(x)
            loss = nn.CrossEntropyLoss(y_predict, y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            y_predict_history.append(y_predict)

    return loss_history, y_predict_history


def test_model(test, net, device=None, lr=1e-3):
    assert device is not None, "device must be cpu or cuda"
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
