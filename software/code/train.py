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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
print(f'cuda version: {torch.version.cuda}')

config = read_config()

train_directory = Path(config['train_directory'])
model_directory = Path(config['model_directory'])
output_directory = Path(config['output_directory'])
model_name = config['model_name']
random_seed = config['random_seed']

num_workers = os.cpu_count()

if random_seed != 0:
    torch.manual_seed(random_seed)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='min',
                                                           patience=config['patience'])
    
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           validation_dataloader=validation_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           epochs=config['epochs'],
                           device=device,
                           model_directory=model_directory,
                           output_directory=output_directory,
                           model_name=model_name,
                           minimum_epochs=config['minimum_epochs'])
    
    loss_curve_path = generate_file_path(model_name,
                                         output_directory,
                                         file_extension='jpg',
                                         directory_name='loss-curve')

    plot_loss_curves(results, loss_curve_path)

if __name__ == '__main__':
    main()