import os
import random
import shutil
import zipfile
from tqdm import tqdm

def extract_zip(zip_file, extract_to):
    """
    Extracts a zip file to the given destination directory.
    """
    print(zip_file)
    print("Extracting {} to {}".format(zip_file, extract_to))
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

    return extract_to

def split_data(source_folder, destination, split_ratio, random_seed):
    """
    Splits the data into train and validation sets.
    """
    random.seed(random_seed)
    print("Splitting data into train and validation sets...")

    source_folder = os.path.join(source_folder, "archive")
    all_files = os.listdir(source_folder)
    random.shuffle(all_files)
    num_train = int(len(all_files) * split_ratio)
    train_files = all_files[:num_train]
    val_files = all_files[num_train:]

    train_folder = os.path.join(destination, "train")
    test_folder = os.path.join(destination, "test")

    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
        print(f"Previous train folder {train_folder} has been removed.")
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
        print(f"Previous validation folder {test_folder} has been removed.")
 
    os.makedirs(train_folder)
    os.makedirs(test_folder)

    for file in tqdm(train_files, desc="Copying train data"):
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))

    for file in tqdm(val_files, desc="Copying validation data"):
        shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))

    print("Data splitting completed.")

    return train_folder, test_folder
