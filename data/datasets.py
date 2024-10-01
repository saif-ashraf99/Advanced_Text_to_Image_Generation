import os
import json
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Load captions
        with open(captions_file, 'r') as f:
            self.id_to_captions = json.load(f)
        
        # Build list of image files and corresponding captions
        self.image_ids = list(self.id_to_captions.keys())
        self.image_files = [os.path.join(image_dir, f"{int(img_id):012d}.jpg") for img_id in self.image_ids]
        self.captions = [self.id_to_captions[img_id][0] for img_id in self.image_ids]  # Using the first caption
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption = self.captions[idx]
        return image, caption
