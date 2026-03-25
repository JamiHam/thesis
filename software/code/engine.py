import os
from pathlib import Path
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from utils import save_model

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    model.train()

    running_loss = 0
    running_accuracy = 0

    for batch, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        predicted_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        loss = loss_function(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_accuracy += (predicted_labels == labels).sum().item() / len(labels)

    train_loss = running_loss / len(dataloader)
    train_accuracy = running_accuracy / len(dataloader)

    return train_loss, train_accuracy

def validation_step(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_function: torch.nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    
    model.eval()

    running_loss = 0
    running_accuracy = 0

    with torch.inference_mode():
        for batch, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            loss = loss_function(outputs, labels)

            running_loss += loss.item()
            running_accuracy += (predicted_labels == labels).sum().item() / len(labels)

    validation_loss = running_loss / len(dataloader)
    validation_accuracy = running_accuracy / len(dataloader)

    return validation_loss, validation_accuracy

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          validation_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          model_directory: Path,
          model_name: str) -> Dict[str, List]:
    
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                device=device)
        
        validation_loss, validation_accuracy = validation_step(model=model,
                                                               dataloader=validation_dataloader,
                                                               loss_function=loss_function,
                                                               device=device)
        
        print(
            f'Epoch: {epoch + 1} | '
            f'train_loss: {train_loss:.4f} | '
            f'train_accuracy: {train_accuracy:.2f}% | '
            f'validation_loss: {validation_loss:.4f} | '
            f'validation_accuracy: {validation_accuracy:.2f}%'
        )

        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['validation_loss'].append(validation_loss)
        results['validation_accuracy'].append(validation_accuracy)

        save_model(model=model,
                   model_directory=model_directory,
                   model_name=model_name,
                   epoch=epoch + 1)

    return results