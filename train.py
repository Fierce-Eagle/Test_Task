import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_model(loader_train, loader_val, net, epochs=10, device=None, lr=1e-3):
    """
    return: потери +, лучшая модель +, предсказанные оценки для валидационного набора
    """
    assert device is not None, "device must be cpu or cuda"
    сrossEntropy = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr)
    loss_history = []  # потери
    y_pred_history = []
    model = net.to(device)
    best_model = None  # лучшая модель
    best_acc = 0
    batch_num = len(loader_train)

    for epoch in range(epochs):
        loss_sum = 0
        current_acc = 0
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.int64)
            y = torch.flatten(y)
            predicted_y = model(x)
            loss = сrossEntropy(predicted_y, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(predicted_y, dim=1)
            correct = pred.eq(y)
            
            y_pred_history.append(np.array(pred))  # предсказанные
            loss_sum += loss.item()
            acc = torch.mean(correct.float())
            current_acc += acc
            if t % 80 == 0:
                
                print('Epoch [%d/%d], Iteration %d, loss = %.4f acc = %.4f' % (epoch, epochs, t, loss.item(), acc))
                
        loss_history.append(loss_sum / batch_num)  
        current_acc /= batch_num
        
        y_pred_valid = test_model(model, loader_val, device)
        
        if current_acc > best_acc:
            best_acc = current_acc
            best_model = model
            best_y_valid = y_pred_valid

    return loss_history, best_model, best_y_valid


def test_model(model, loader_test, device=None):
    assert device is not None, "device must be cpu or cuda"
    model.eval()
    predict_list = []

    with torch.no_grad():
        for x, _ in loader_test:
            x = x.to(device=device, dtype=torch.float32)
            scores = model(x).round()
            pred = torch.argmax(scores, dim=1)
            predict_list += [p.item() for p in pred]

    return np.array(predict_list)