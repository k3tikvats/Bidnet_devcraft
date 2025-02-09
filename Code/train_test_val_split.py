import random
from tqdm import tqdm

def split_data(input_file, train_ratio=0.85, test_ratio=0.10, val_ratio=0.05):
    """
    Splits a text file into train, test, and validation sets based on given ratios.
    :param input_file: Path to the input text file.
    :param train_ratio: Ratio of data for training set.
    :param test_ratio: Ratio of data for testing set.
    :param val_ratio: Ratio of data for validation set.
    """
    assert train_ratio + test_ratio + val_ratio == 1, "Ratios must sum to 1."
    
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Shuffle the lines randomly
    random.shuffle(lines)
    
    # Create empty lists for train, test, and validation data
    train_data, test_data, val_data = [], [], []
    
    # Assign lines to train, test, or val using random selection
    for line in tqdm(lines):
        rand_val = random.random()
        if rand_val < train_ratio:
            train_data.append(line)
        elif rand_val < train_ratio + test_ratio:
            test_data.append(line)
        else:
            val_data.append(line)
    
    # Save to separate files
    for split_name, data in zip(["train12.txt", "test12.txt", "val12.txt"], [train_data, test_data, val_data]):
        with open(split_name, 'w', encoding='utf-8') as f:
            f.writelines(data)
        print(f"Saved {len(data)} lines to {split_name}")

# Example Usage
if __name__ == "__main__":
    split_data("/home/akshit/adobe devcraft/Adobe Devcraft Dataset/dataset/imp.12.txt")  # Replace 'input.txt' with your actual file path
