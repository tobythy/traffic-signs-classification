import argparse
import torch
import torchvision.datasets as datasets

from data import data_transforms
from model import Net

TEST_DIR = 'traffic-sign/test'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='PyTorch evaluation')

parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")

args = parser.parse_args()


state_dict = torch.load(args.model)
model = Net()
model.load_state_dict(state_dict)
model.eval()


eval_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(TEST_DIR, transform=data_transforms))


total, correct = 0, 0
for img, label in eval_loader:
    images = img.to(device)
    labels = label.to(device)
    output = model(images)
    _, idx = torch.max(output.data, 1)
    total += labels.size(0)

    if idx == labels:
        correct += 1

print(f"accuracy:{correct / total}")
# model_40.pth accuracy:0.9825396825396825