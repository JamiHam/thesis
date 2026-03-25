import os
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import configparser
from torchinfo import summary

def read_config():
    config = configparser.ConfigParser()
    config.read('software/code/config.ini')

    train_directory = config.get('Path', 'train_directory')
    test_directory = config.get('Path', 'test_directory')
    model_directory = config.get('Path', 'model_directory')
    output_directory = config.get('Path', 'output_directory')
    grad_cam_image_path = config.get('Path', 'grad_cam_image_path')

    model = config.get('Model', 'model')
    pretrained = config.getboolean('Model', 'pretrained')
    size = config.get('Model', 'size')
    resnet_layers = config.getint('Model', 'resnet_layers')
    resnet_version = config.getint('Model', 'resnet_version')
    swin_transformer_version = config.getint('Model', 'swin_transformer_version')
    model_name = config.get('Model', 'model_name')
    target_epoch = config.getint('Model', 'target_epoch')

    batch_size = config.getint('Training', 'batch_size')
    epochs = config.getint('Training', 'epochs')
    learning_rate = config.getfloat('Training', 'learning_rate')

    config_values = {
        'train_directory': train_directory,
        'test_directory': test_directory,
        'model_directory': model_directory,
        'output_directory': output_directory,
        'grad_cam_image_path': grad_cam_image_path,
        'model': model,
        'pretrained': pretrained,
        'size': size,
        'resnet_layers': resnet_layers,
        'resnet_version': resnet_version,
        'swin_transformer_version': swin_transformer_version,
        'model_name': model_name,
        'target_epoch': target_epoch,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate
    }

    return config_values
    
def generate_file_path(model_name: str,
                       output_directory: Path,
                       file_extension: str,
                       directory_name: str) -> Path:
    
    target_directory = output_directory / directory_name
    
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    #file_name = model_name.split('.')[0]
    file_name = model_name
    
    i = 0
    while os.path.exists(f'{target_directory}/{file_name}-{i}.{file_extension}'):
        i += 1

    return f'{target_directory}/{file_name}-{i}.{file_extension}'

def save_model(model: torch.nn.Module,
               model_directory: Path,
               model_name: str,
               epoch: int):
    
    target_directory = f"{model_directory}/{model_name}"
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    model_path = f"{target_directory}/{model_name}_epoch{epoch}.pth"
    torch.save(obj=model.state_dict(),
               f=model_path)

def create_model_summary(model: nn.Module) -> str:
    return summary(model=model,
                   input_size=(32, 3, 224, 224),
                   col_names=['input_size', 'output_size', 'num_params', 'trainable'],
                   col_width=20,
                   row_settings=['var_names'])