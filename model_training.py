import torch
import pandas as pd

from foundry.transforms import Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

street_images = Dataset.get("street_images").files().download()
geo_data = Dataset.get("geo_data").read_table(format="pandas")

us_geo_data = geo_data.iloc[:8_000 + 1, ].copy()
us_geo_data['full_path'] = list(street_images.values())

us_geo_data = us_geo_data[['image_path', 'full_path', 'latitude', 'longitude']]
us_geo_data.rename(columns={'image_path': 'image_name'}, inplace=True)

class StreetImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing metadata (image path, latitude, longitude).
            image_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on the image.
        """
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get image metadata
        row = self.dataframe.iloc[idx]
        image_path = row['full_path']
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        
        # You can return additional metadata, such as latitude, longitude, etc.
        latitude = row['latitude']
        longitude = row['longitude']
        
        return image, latitude, longitude
