from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
def get_transforms():
    train_transform = transforms.Compose([
        # transforms.CenterCrop(224),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_loaders(dataset_dir, batch_size=32, num_workers=4):
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Define train and validation directories
    train_dir = f"{dataset_dir}/train"
    val_dir = f"{dataset_dir}/val"
    
    # Load datasets
    train_data = datasets.ImageFolder(train_dir)
    val_data = datasets.ImageFolder(val_dir)
    
    # Apply transforms
    train_data.transform = train_transform
    val_data.transform = val_transform
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, val_loader