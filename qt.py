import torch
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torchvision import datasets, transforms
from networks.resnet_big import *

model_path = "/content/"
ds_path = "/content/EmojiDataset"

normalize = transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.25, 0.25, 0.25))
qt_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.25, 0.25, 0.25))
])

q_config = get_default_qconfig()
qconfig_dict = {"": q_config}

ds = datasets.ImageFolder(root = ds_path, transform = qt_transform)
loader = torch.utils.data.DataLoader(ds, batch_size = 16, num_workers = 4, pin_memory = True, sampler = None)

model = torch.load(model_path)
model.eval()

def calibrate(model, data_loader, num = 256):
    model.eval()
    with torch.no_grad():
        i = 0
        for image, _ in data_loader:
            model(image)
            i += 1
            if i >= num: break

p_model = prepare_fx(model, qconfig_dict, example_inputs = torch.randn(16, 3, 224, 224)).cpu()
calibrate(p_model, loader)
q_model = convert_fx(p_model)

torch.save(q_model.state_dict(), model_path + ".qt.pth")