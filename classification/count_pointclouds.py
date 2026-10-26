import os
from classification import PointCloudDataset  # Adjust this import to where your PointCloudDataset class is defined

def count_samples(root_dir):
    """
    Instantiate the dataset for the given root directory and return the number of samples.
    """
    dataset = PointCloudDataset(root_dir=root_dir)
    return len(dataset)

def main():
    # These directories should point to your train, validation, and test sets respectively.
    train_dir = '/data/seankirkby/data/train'
    val_dir   = '/data/seankirkby/data/val'
    test_dir  = '/data/seankirkby/data/test'
    
    train_count = count_samples(train_dir)
    val_count   = count_samples(val_dir)
    test_count  = count_samples(test_dir)
    
    print(f"Training set contains {train_count} point cloud samples.")
    print(f"Validation set contains {val_count} point cloud samples.")
    print(f"Test set contains {test_count} point cloud samples.")

if __name__ == "__main__":
    main()
