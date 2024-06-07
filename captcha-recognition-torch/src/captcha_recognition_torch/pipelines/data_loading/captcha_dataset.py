import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CAPTCHADataset(Dataset):
    def __init__(self, data_path):
        self._data_path = data_path
        
    def __len__(self):
        return len(os.listdir(self._data_path))
    
    def __getitem__(self, index):
        image_fn = os.listdir(self._data_path)[index]
        image_fp = os.path.join(self._data_path, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)
        text = image_fn.split(".")[0]
        return image, text
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transform_ops(image)