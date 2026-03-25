from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import model_setup
from utils import read_config, generate_file_path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = read_config()

model_directory = Path(config['model_directory'])
data_directory = Path(config['test_directory'])
image_path = Path(config['grad_cam_image_path'])
output_directory = Path(config['output_directory'])

model_type = config['model']
model_name = config['model_name']
target_epoch = config['target_epoch']

def predict(model, image_tensor, class_names):
    model.eval()

    with torch.inference_mode():
        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1)
        
        confidence = probabilities[0][predicted_label].item()

        results = {
            'predicted_label': class_names[predicted_label],
            'confidence': f'{confidence * 100:.2f}%',
        }

        return results

# Get the last convolutional layer of the model
def get_target_layer(model, model_type):
    if model_type == 'convnext':
        return model.features[-1]
    
    if model_type == 'resnet':
        return model.layer4
    
    if model_type == 'swin_transformer':
        return model.features[-1][-1].norm2

def create_cam_image(model, image, image_tensor):
    target_layers = [get_target_layer(model, model_type)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    return cam_image

def display_cam_image(cam_image, title):
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.imread(image_path)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(original_image)
    ax[0].axis('off')
    ax[1].imshow(cam_image)
    ax[1].axis('off')

    fig.tight_layout()
    fig.suptitle(title)

    save_figure(plt)
    plt.show()

def save_figure(figure):
    file_path = generate_file_path(model_name,
                                   output_directory,
                                   file_extension='jpg',
                                   directory_name='grad-cam')
    
    figure.savefig(file_path, bbox_inches='tight', pad_inches=0.1)

def main():
    class_names = datasets.ImageFolder(data_directory).classes
    model, preprocess = model_setup.setup(model=model_type,
                                          pretrained=False,
                                          class_names=class_names,
                                          device=device,
                                          size=config['size'],
                                          resnet_layers=config['resnet_layers'],
                                          swin_transformer_version=config['swin_transformer_version'])
    
    model_path = f"{model_directory}/{model_name}/{model_name}_epoch{target_epoch}.pth"
    model.load_state_dict(torch.load(f=model_path,
                                     map_location=device))
    
    image = cv2.imread(image_path, 1)[:, :, ::-1]
    image = np.float32(image) / 255
    image_tensor = preprocess_image(image,
                                    mean=preprocess.mean,
                                    std=preprocess.std).to(device)
    
    results = predict(model, image_tensor, class_names)
    title = f'Predicted label: {results['predicted_label']}\nConfidence {results['confidence']}'
    
    cam_image = create_cam_image(model, image, image_tensor)
    display_cam_image(cam_image, title)

if __name__ == '__main__':
    main()