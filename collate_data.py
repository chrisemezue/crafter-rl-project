import os
import pickle
import numpy as np
import torch
import re
import shutil

# Specify the folder path where you want to search for files
folder_path = "/home/mila/c/connor.brennan/scratch/crafter-attn"
output_dir = "/home/mila/c/connor.brennan/scratch/crafter-curated"


def doFileAction(file_path):
    # Get the base name of the file
    file_name = os.path.basename(file_path)

    pattern = re.compile(r'.*attn_patch_(\d+).*')
    match = pattern.search(file_path)
    patchSize = int(match.group(1))

    pattern = re.compile(r'.*_(\d+)')
    match = pattern.search(file_path)
    timesteps = int(match.group(1))

    pattern = re.compile(r'(.*)_\d+')
    match = pattern.search(file_name)
    name = match.group(1)

    runType = 'valid_sto'
    if 'valid_det' in file_path:
        runType = 'valid_det'

    destination_folder = f'{output_dir}/t_{timesteps}/'
    save_name = f'patch_size_{patchSize}_{runType}_{name}'

    print(f'{destination_folder}{save_name}')

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Move the file to the destination folder while preserving subfolders
    shutil.copy(file_path, os.path.join(destination_folder, save_name))

    # Perform action on file
    '''with open(file_path, 'rb') as file:
        print("Processing file:", file_object.name)'''

def applyFilter(file_name, pattern):
    regex = re.compile(pattern)

    # Define your filter logic here
    if regex.search(file_name):
        return True
    else:
        return False

def find_files(folder_path, regex):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if applyFilter(file_path, regex):
                doFileAction(file_path)
                    


def variable_summary(variable, indent=''):
    print(f"{indent}Summary of the variable:")

    print(f"{indent}Type:", type(variable))

    if isinstance(variable, (list, tuple, np.ndarray, torch.Tensor)):
        print(f"{indent}Shape:", variable.shape)

    if isinstance(variable, (list, tuple)):
        print(f"{indent}Length:", len(variable))

        for index, item in enumerate(variable):
            print(f"{indent}Element {index}:")
            variable_summary(item, indent + '  ')

    if isinstance(variable, dict):
        print(f"{indent}Keys:", variable.keys())

        for key, value in variable.items():
            print(f"{indent}Key '{key}':")
            variable_summary(value, indent + '  ')

    if isinstance(variable, (int, float, np.number, torch.Tensor)):
        print(f"{indent}Value:", variable)

    print(f"{indent}Representation:", repr(variable))


if __name__ == '__main__':
    for timestep in [50000, 150000, 350000, 850000, 125000, 2050000, 4050000]:
        # Call the function to find and process files
        find_files(folder_path, f'.*_{timestep}$')

    '''with open(f'{folder_path}/attn_patch_16_stride_16/resilient-sound-53/valid_det/attn_maps_50000', 'rb') as file:
        data = pickle.load(file)

        variable_summary(data)'''
