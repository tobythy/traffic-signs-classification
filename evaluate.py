import argparse
import torch
import torchvision.datasets as datasets

from data import data_transforms
from model import Net

TEST_DIR = 'traffic-sign/test'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='PyTorch 62 Traffic Sign Recognition  evaluation')

parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='pred.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()


state_dict = torch.load(args.model)
model = Net()
model.load_state_dict(state_dict)
model.eval()

output_file = open(args.outfile, "w")
output_file.write("filename,label,predict,match\n")

eval_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(TEST_DIR, transform=data_transforms))


total, correct = 0, 0
for i, (images, labels) in enumerate(eval_loader, 0):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    sample_fname, _ = eval_loader.dataset.samples[i]
    total += labels.size(0)

    if labels == predicted:
        match = 1
        correct += 1
    else:
        match = 0

    output_file.write("%a,%s,%d,%f\n" % (sample_fname, labels.item(), predicted.item(), match))

    if i % 100 == 0:
        print("{}: {}, predict:{}, label:{}, accuracy:{}\n".format(i, sample_fname, predicted.item(), labels.item(), correct / total))

output_file.close()

print(f"accuracy:{correct / total}")