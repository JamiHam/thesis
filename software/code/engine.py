import os
from pathlib import Path
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def save_training_results(output_directory: Path,
                          model_name: str,
                          results: Dict,
                          epoch: int):
    
    results_string = (
        f'Epoch: {epoch} | '
        f'train_loss: {results["train_loss"][-1]:.4f} | '
        f'train_accuracy: {results["train_accuracy"][-1]:.2f}% | '
        f'validation_loss: {results["validation_loss"][-1]:.4f} | '
        f'validation_accuracy: {results["validation_accuracy"][-1]:.2f}%\n'
    )

    target_directory = output_directory / 'training-results'
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    file_path = target_directory / model_name
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(results_string)

    return results_string

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
          output_directory: Path,
          model_name: str) -> Dict[str, List]:
    
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': []
    }

    lowest_validation_loss = 1

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

        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['validation_loss'].append(validation_loss)
        results['validation_accuracy'].append(validation_accuracy)

        print(save_training_results(output_directory=output_directory,
                                    model_name=model_name,
                                    results=results,
                                    epoch=epoch + 1))
        
        if validation_loss < lowest_validation_loss:
            print(f'Lowest validation loss so far ({validation_loss:.4f}), saving model...')

            model_path = model_directory / model_name
            torch.save(obj=model.state_dict(),
                       f=model_path)
            
            lowest_validation_loss = validation_loss

    return results