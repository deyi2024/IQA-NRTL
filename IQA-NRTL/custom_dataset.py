import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = float(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # Example usage
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(csv_file='train_labels.csv', root_dir='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Iterate through the dataset
    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break
