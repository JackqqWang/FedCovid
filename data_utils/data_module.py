import torch
from torch.utils.data.dataset import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
          assert len(inputs) == len(labels)
          self.inputs = torch.tensor(inputs)
          self.labels = torch.tensor(labels).long()
          self.transforms = transforms

    def __getitem__(self, index):
          img, label = self.inputs[index], self.labels[index]

          if self.transforms is not None:
            img = self.transforms(img)

          return (img, label)

    def __len__(self):
          return len(self.inputs)



