from hebb_net import HebbRuleWithActivationThreshold, gradiant_sparsity
import torch


def train(dataloader, model, device, p, loss_fn, optimizer, lr, activation_thresholder: HebbRuleWithActivationThreshold):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X,z_hidden,pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # optimize classifiction wieghts
        optimizer.step()

        # optimize hebbian weights
        # activation threshold
        delta_w1 = activation_thresholder(X, z_hidden)

        # Gradient sparsity
        delta_w1 = gradiant_sparsity(delta_w1, p, device)

        # update hebbian weights
        model.hebbian_weights.weight.data = model.hebbian_weights.weight.data - lr*delta_w1

        if batch % 10000 == 0:
          loss, current = loss.item(), (batch + 1) * len(X)
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    print(f"Train Accuracy: {(100*correct):>0.1f}%")


def test(dataloader, model, device,  loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            x,z,pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100* correct