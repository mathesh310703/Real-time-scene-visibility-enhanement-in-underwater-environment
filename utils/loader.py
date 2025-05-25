from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class CustomLoader(Dataset):
    def __init__(self, distorted_list_file_path, restored_list_file_path, dataset_root):
        distorted_list_file = open(distorted_list_file_path, 'r')
        self.distorted_list = distorted_list_file.readlines()
        restored_list_file = open(restored_list_file_path, 'r')
        self.restored_list = restored_list_file.readlines()
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([
                transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.distorted_list)

    def __getitem__(self, idx):
        distorted_frame = Image.open(os.path.join(
            self.dataset_root, self.distorted_list[idx][0:-1])).convert('RGB')
        restored_frame = Image.open(os.path.join(
            self.dataset_root, self.restored_list[idx][0:-1])).convert('RGB')

        if self.transform:
            distorted_frame = self.transform(distorted_frame)
            restored_frame = self.transform(restored_frame)

        return {
            # Used to check whether the frame belongs to one video
            'name': self.restored_list[idx][0:-1].split("/")[-2],
            'distorted_frame': distorted_frame, 
            'restored_frame': restored_frame
        }
