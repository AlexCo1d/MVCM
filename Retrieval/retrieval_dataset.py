import pandas as pd
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from pretrain_datasets import pre_caption


class retrieval_dataset(Dataset):
    def __init__(self, data_path, task:str = 'retrieval', args=None):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data = pd.read_csv(os.path.join(data_path, 'df_200.csv'))
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.data_path, sample['Path'])
        img = Image.open(img_path).convert('RGB')
        if self.task == 'retrieval':
            report = sample['report_content']
        else:
            report = sample['Class']
        report = pre_caption(report, 256)
        item = {
            'image': self.transform(img),
            'text': report,
        }
        return item

class retrieval_dataset_ROCO(Dataset):
    def __init__(self, data_path, args):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data = pd.read_csv(os.path.join(data_path, 'test.csv'))
        # use the first 2000 data sample, as m3ae and arl
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)[:2000]
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.data_path, 'images', sample['image_path'])
        img = Image.open(img_path).convert('RGB')
        report = sample['text']

        report = pre_caption(report, 256)
        item = {
            'image': self.transform(img),
            'text': report,
        }
        return item
