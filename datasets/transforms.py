from torchvision import transforms

def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.ToTensor()
        ])
    elif split in ('val', 'validation'):
        return transforms.Compose([
            transforms.ToTensor()
        ])
    elif split == 'test':
        return transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Unsupported split: {split}")
