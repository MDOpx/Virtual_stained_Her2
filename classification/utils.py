import os
import re


def extract_class_from_filename(filename):
    """Extract class label from filename like '00000_train_1+.png'"""
    match = re.search(r'_(\d\+?)(?=\.png)', filename)
    if match:
        class_str = match.group(1)
        if class_str == '0':
            return 0
        elif class_str == '1+':
            return 1
        elif class_str == '2+':
            return 2
        elif class_str == '3+':
            return 3
    raise ValueError(f"Could not extract class from filename: {filename}")


def load_single_data(data_dir):
    """Load single image data from directory"""
    image_paths = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(data_dir, filename)
            try:
                label = extract_class_from_filename(filename)
                image_paths.append(img_path)
                labels.append(label)
            except ValueError as e:
                print(f"Warning: {e}, skipping file")
                continue

    return image_paths, labels


def load_multi_data(data_dirA, data_dirB):
    """Load paired image data from two directories"""
    filesA = {f: os.path.join(data_dirA, f) for f in os.listdir(data_dirA) if f.endswith('.png')}
    filesB = {f: os.path.join(data_dirB, f) for f in os.listdir(data_dirB) if f.endswith('.png')}
    common_files = set(filesA.keys()) & set(filesB.keys())
    image_pairs = []
    labels = []
    for filename in common_files:
        try:
            label = extract_class_from_filename(filename)
            image_pairs.append((filesA[filename], filesB[filename]))
            labels.append(label)
        except ValueError as e:
            print(f"Warning: {e}, skipping file")
            continue
    return image_pairs, labels
