import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import datasets
from typing import Dict, List

import engine
import model_setup
from utils import read_config, generate_file_path

config = read_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = os.cpu_count()

train_directory = Path(config['train_directory'])
model_directory = Path(config['model_directory'])
output_directory = Path(config['output_directory'])
model_name = config['model_file_name']

def plot_loss_curves(results: Dict[str, List[float]],
                    loss_curve_path: Path):
    
    train_loss = results['train_loss']
    validation_loss = results['validation_loss']
    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, validation_loss, label='validation_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(loss_curve_path, bbox_inches='tight', pad_inches=0.1)

def save_model(model: nn.Module,
               directory: Path,
               model_name: str):
    
    model_path = directory / model_name
    torch.save(obj=model.state_dict(),
               f=model_path)
    
def main():
    class_names = datasets.ImageFolder(train_directory).classes
    model, preprocess = model_setup.setup(model=config['model'],
                                          pretrained=config['pretrained'],
                                          class_names=class_names,
                                          device=device,
                                          size=config['size'],
                                          resnet_layers=config['resnet_layers'],
                                          resnet_version=config['resnet_version'],
                                          swin_transformer_version=config['swin_transformer_version'])

    dataset = datasets.ImageFolder(train_directory, transform=preprocess)
    train_set, validation_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   pin_memory=True)
    
    validation_dataloader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=config['batch_size'],
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        pin_memory=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           validation_dataloader=validation_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           epochs=config['epochs'],
                           device=device)
    
    loss_curve_path = generate_file_path(model_name,
                                         output_directory,
                                         file_extension='jpg',
                                         directory_name='loss-curve')

    plot_loss_curves(results, loss_curve_path)
    save_model(model, model_directory, model_name)

if __name__ == '__main__':
    main()