from typing import Dict, List
import torch
from torchvision import datasets
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import model_setup
from utils import read_config, generate_file_path

config = read_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_directory = Path(config['test_directory'])
model_directory = Path(config['model_directory'])
output_directory = Path(config['output_directory'])
model_name = config['model_file_name']

def calculate_metrics(true_labels: List,
                      predicted_labels: List,
                      label_indexes: List) -> Dict:
    metrics = {}

    metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
    metrics['precision'] = precision_score(true_labels, predicted_labels, labels=label_indexes, average=None)
    metrics['recall'] = recall_score(true_labels, predicted_labels, labels=label_indexes, average=None)
    metrics['f1'] = f1_score(true_labels, predicted_labels, labels=label_indexes, average=None)

    return metrics

def create_confusion_matrix(true_labels: List,
                            predicted_labels: List,
                            class_names: List) -> plt.Figure:
    
    matrix = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    return display.plot(cmap=plt.cm.Blues).figure_

def save_metrics(metrics: Dict,
                 class_names: List,
                 file_path: Path):

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f'Overall accuracy: {metrics['accuracy']}')

        for i in range(len(class_names)):
            class_name = class_names[i]

            file.write(f'\n\nMetrics for "{class_name}" class:\n')
            file.write(f'Precision: {metrics['precision'][i]}\n')
            file.write(f'Recall: {metrics['recall'][i]}\n')
            file.write(f'F1 Score: {metrics['f1'][i]}')

    with open(file_path, 'r', encoding='utf-8') as file:
        print(file.read())

def main():
    class_names = datasets.ImageFolder(test_directory).classes

    model, preprocess = model_setup.setup(model=config['model'],
                                          pretrained=False,
                                          class_names=class_names,
                                          device=device,
                                          size=config['size'],
                                          resnet_layers=config['resnet_layers'],
                                          swin_transformer_version=config['swin_transformer_version'])

    model.load_state_dict(torch.load(f=model_directory / model_name,
                                     map_location=device))

    test_set = datasets.ImageFolder(test_directory, transform=preprocess)
    test_dataloader = torch.utils.data.DataLoader(test_set)

    model.eval()

    label_indexes = []
    for i in range(len(class_names)):
        label_indexes.append(i)

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        for input, label in tqdm(test_dataloader):
            input, label = input.to(device), label.to(device)

            output = model(input)
            predicted_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            
            true_labels.append(label.item())
            predicted_labels.append(predicted_label.item())

        metrics = calculate_metrics(true_labels,
                                    predicted_labels,
                                    label_indexes)
        
        metrics_path = generate_file_path(model_name,
                                          output_directory,
                                          file_extension='txt',
                                          directory_name='metrics')
        
        save_metrics(metrics,
                     class_names,
                     file_path=metrics_path)
        
        cm = create_confusion_matrix(true_labels,
                                     predicted_labels,
                                     class_names)
        
        cm_path = generate_file_path(model_name,
                                     output_directory,
                                     file_extension='jpg',
                                     directory_name='confusion-matrix')
        
        cm.savefig(cm_path, bbox_inches='tight', pad_inches=0.1)
        cm.show()

if __name__ == '__main__':
    main()