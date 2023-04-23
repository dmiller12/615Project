import torch
import tqdm
from torchvision import transforms

from models.C4CNN import C4CNN
from models.C8SteerableCNN import C8SteerableCNN
from models.C4SteerableCNN import C4SteerableCNN
from models.StandardCNN import StandardCNN
from models.SO2SteerableCNN import SO2SteerableCNN
from data.dataset import MnistRotDataset


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
])

test_set = MnistRotDataset('test', transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
total = 0
correct = 0

model_list = [
    StandardCNN,
    C4CNN,
    C4SteerableCNN,
    C8SteerableCNN,
    SO2SteerableCNN,
]

for mod_class in model_list:
    model = mod_class(n_channels=3)
    params = torch.load(f"saved_models/{model.__class__.__name__}_{test_set.__class__.__name__}.pt", map_location=torch.device(device))
    model.load_state_dict(params)
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
    print(f"{model.__class__.__name__} Test Accuracy: {accuracy}")