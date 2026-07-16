import os
from pathlib import Path
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from timeit import default_timer as timer
import time
from utils import save_to_file

def save_epoch_results(output_directory: Path,
                       model_name: str,
                       results: Dict,
                       epoch: int,
                       learning_rate: float):
    
    formatted_learning_rate = str(learning_rate).rstrip('0')
    
    results_string = (
        f'Epoch: {epoch} | '
        f'train_loss: {results["train_loss"][-1]:.4f} | '
        f'train_accuracy: {results["train_accuracy"][-1]:.2f}% | '
        f'validation_loss: {results["validation_loss"][-1]:.4f} | '
        f'validation_accuracy: {results["validation_accuracy"][-1]:.2f}% | '
        f'learning_rate: {formatted_learning_rate}'
    )

    directory = output_directory / 'training-results'
    file_name = f'{model_name}.txt'
    
    save_to_file(directory=directory,
                 file_name=file_name,
                 content=results_string)

    return results_string

def save_training_results(output_directory: Path,
                          model_name: str,
                          start_time: float,
                          end_time: float,
                          epochs: int,
                          lowest_validation_loss: float,
                          best_epoch: int):
    
    total_seconds = end_time - start_time
    total_time = time.strftime('%H:%M:%S', time.gmtime(total_seconds))
    seconds_per_epoch = total_seconds / epochs

    results_string = (
        f'Total training time: {total_time} | '
        f'Time per epoch: {seconds_per_epoch:.2f} seconds | '
        f'Lowest validation loss: {lowest_validation_loss:.4f} | '
        f'Best epoch: {best_epoch}'
    )

    directory = output_directory / 'training-results'
    file_name = f'{model_name}.txt'

    save_to_file(directory=directory,
                 file_name=file_name,
                 content=results_string)
    
    return results_string

def save_model(model: torch.nn.Module,
               model_name: str,
               model_directory: Path):
    
    model_path = f'{model_directory}/{model_name}.pth'
    torch.save(obj=model.state_dict(),
                f=model_path)

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
          scheduler: torch.optim.lr_scheduler,
          epochs: int,
          device: torch.device,
          model_directory: Path,
          output_directory: Path,
          model_name: str,
          minimum_epochs: int) -> Dict[str, List]:
    
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': []
    }

    lowest_validation_loss = 1
    best_epoch = 0

    start_time = timer()

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                device=device)
        
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        
        validation_loss, validation_accuracy = validation_step(model=model,
                                                               dataloader=validation_dataloader,
                                                               loss_function=loss_function,
                                                               device=device)
        
        results['validation_loss'].append(validation_loss)
        results['validation_accuracy'].append(validation_accuracy)
        
        if epoch >= minimum_epochs:
            scheduler.step(validation_loss)

        print(save_epoch_results(output_directory=output_directory,
                                 model_name=model_name,
                                 results=results,
                                 epoch=epoch + 1,
                                 learning_rate=optimizer.param_groups[0]['lr']))
        
        if validation_loss < lowest_validation_loss:
            print(f'Lowest validation loss so far ({validation_loss:.4f}), saving model...')

            save_model(model=model,
                       model_name=model_name,
                       model_directory=model_directory)
            
            lowest_validation_loss = validation_loss
            best_epoch = epoch + 1

    end_time = timer()

    print(save_training_results(output_directory=output_directory,
                                model_name=model_name,
                                start_time=start_time,
                                end_time=end_time,
                                epochs=epochs,
                                lowest_validation_loss=lowest_validation_loss,
                                best_epoch=best_epoch))

    return results