from torchvision import transforms

def get_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return test_transform
from torchvision import datasets
from torch.utils.data import DataLoader

def get_test_loader(dataset_dir, batch_size=32, num_workers=4):
    # Get the test transform
    test_transform = get_test_transform()
    
    # Define the test directory
    test_dir = f"{dataset_dir}/test"
    
    # Load the test dataset
    test_data = datasets.ImageFolder(test_dir)
    # Apply the transform
    test_data.transform = test_transform
    
    file_names = ["/Class%s/"%_ +  path.split("/")[-2] + path.split("/")[-1] for path, _ in test_data.samples]
    # Create the DataLoader
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return test_loader,file_names


