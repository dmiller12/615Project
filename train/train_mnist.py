import torch
import tqdm
from torchvision.transforms import ToTensor
from torchvision import transforms
from models.C4CNN import C4CNN
from models.C8SteerableCNN import C8SteerableCNN
from models.C4SteerableCNN import C4SteerableCNN
from models.StandardCNN import StandardCNN
from models.SO2SteerableCNN import SO2SteerableCNN
import numpy as np
import time
from data.dataset import MnistRotDataset



def train_model(model: torch.nn.Module, train_set, val_set, device):

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    model.to(device)
    model.train()

    accuracy_list = []
    best = 0
    epoch_since_best = 0

    for epoch in tqdm.tqdm(range(300)):
        model.train()
        total = 0
        correct = 0

        for i, (x, t) in enumerate(train_loader):

            optimizer.zero_grad()

            x = x.to(device)
            t = t.to(device)

            y = model(x)

            loss = loss_function(y, t)

            loss.backward()

            optimizer.step()

            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()

        accuracy = correct / total * 100.0
        print(f"epoch {epoch}, train accuracy: {accuracy}")
        with torch.no_grad():
            model.eval()
            total = 0
            correct = 0
            for j, (x, t) in tqdm.tqdm(enumerate(val_loader)):

                x = x.to(device)
                t = t.to(device)

                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
        accuracy = correct / total * 100.0
        print(f"epoch {epoch}, accuracy: {accuracy}")
        accuracy_list.append(accuracy)
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), f"saved_models/{model.__class__.__name__}_{train_set.__class__.__name__}.pt")
            epoch_since_best = 0
        
        epoch_since_best += 1

        if epoch_since_best > 30:
            return model, accuracy_list

    return model, accuracy_list


def test_model(model: torch.nn.Module, test_set, device):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    total = 0
    correct = 0
    model.to(device)
    with torch.no_grad():
        model.eval()
        for i, (x, t) in tqdm.tqdm(enumerate(test_loader)):

            x = x.to(device)
            t = t.to(device)

            y = model(x)

            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()
    accuracy = correct / total * 100.0

    return accuracy


if __name__ == "__main__":
    dataClass = MnistRotDataset

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = dataClass('train', transform)
    val_set = dataClass('val', transform)
    test_set = dataClass('test', transform)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    ex_image, ex_label = train_set[0]

    # change model here, options found in models/
    model = C4SteerableCNN(n_classes=10, n_channels=ex_image.shape[0])


    print(f"{model.__class__.__name__} on {train_set.__class__.__name__}")
    t1_start = time.perf_counter()
    model, accuracy_list = train_model(model, train_set, val_set, device)
    t1_stop = time.perf_counter()
    print(f"training time: {t1_stop - t1_start} s")

    t2_start = time.perf_counter()
    acc = test_model(model, test_set, device)
    t2_stop = time.perf_counter()
    print(f"inference time: {t2_stop - t2_start} s")
    print(f'Test Accuracy: {acc :.3f}')
    acc_array = np.array(accuracy_list)

    np.save(f"saved_models/{model.__class__.__name__}_{train_set.__class__.__name__}_val_acc.npy", acc_array)
    
    