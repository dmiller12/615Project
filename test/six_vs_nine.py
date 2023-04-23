import torch
import tqdm
from torchvision.transforms import ToTensor
import numpy as np

from models.C4CNN import C4CNN
from models.C8SteerableCNN import C8SteerableCNN
from models.C4SteerableCNN import C4SteerableCNN
from models.StandardCNN import StandardCNN
from models.SO2SteerableCNN import SO2SteerableCNN
from data.dataset import MnistRotDataset, FilteredMNIST

if __name__ == "__main__":

    model_list = [
        StandardCNN,
        C4CNN,
        C4SteerableCNN,
        C8SteerableCNN,
        SO2SteerableCNN,
    ]

    for mod_class in model_list:
        model = mod_class()
        params = torch.load(f"saved_models/{model.__class__.__name__}.pt", map_location=torch.device('cpu'))
        model.load_state_dict(params)

        test_set = MnistRotDataset('test', ToTensor())

        only_9 = FilteredMNIST(test_set, 9)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        test_loader = torch.utils.data.DataLoader(only_9, batch_size=64)
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
        print(f"{model.__class__.__name__} 9 accuracy: {accuracy}")

        only_6 = FilteredMNIST(test_set, 6)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        test_loader = torch.utils.data.DataLoader(only_6, batch_size=64)
        total = 0
        correct = 0
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
        print(f"{model.__class__.__name__} 6 accuracy: {accuracy}")


    
    