import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class split_white_and_gray():
    def __init__(self,threshold=120) -> None:
        """
        Initialize the class with a threshold value.

        Args:
            threshold (int, optional): The threshold value to be set. Defaults to 120.
        """
        self.threshold = threshold

    def __call__(self,tensor):
        """
        Apply thresholding to the input tensor and return the white matter, gray matter, and the original tensor.
        
        Parameters:
        tensor (torch.Tensor): The input tensor to be thresholded.
        
        Returns:
        torch.Tensor: The thresholded white matter.
        torch.Tensor: The thresholded gray matter.
        torch.Tensor: The original input tensor.
        """
        tensor = (tensor*255).to(torch.int64)

        # Apply thresholding
        white_matter = torch.where(tensor >= self.threshold,tensor,0)
        white_matter = (white_matter/255).to(torch.float64)
        gray_matter = torch.where(tensor < self.threshold,tensor,0)
        gray_matter = (gray_matter/255).to(torch.float64)
        tensor = (tensor/255).to(torch.float64)

        return white_matter, gray_matter,tensor
    
def showcam_withoutmask(original_image, grayscale_cam, image_title='Original Image'):
    """This function applies the CAM mask to the original image and returns the Matplotlib Figure object.
    
    :param original_image: The original image tensor in PyTorch format.
    :param grayscale_cam: The CAM mask tensor in PyTorch format.

    :return: Matplotlib Figure object.
    """
    # Assuming you have two tensors: 'original_image' and 'cam_mask'
    # Make sure both tensors are on the CPU
    original_image = torch.squeeze(original_image).cpu()  # torch.Size([3, 150, 150])
    cam_mask = grayscale_cam.cpu()  # torch.Size([1, 150, 150])

    # Convert the tensors to NumPy arrays
    original_image_np = original_image.numpy()
    cam_mask_np = cam_mask.numpy()

    # Apply the mask to the original image
    masked_image = original_image_np * cam_mask_np

    # Normalize the masked_image
    masked_image_norm = (masked_image - np.min(masked_image)) / (np.max(masked_image) - np.min(masked_image))

    # Create Matplotlib Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot the original image
    axes[0].imshow(original_image_np.transpose(1, 2, 0))  # Assuming your original image is in (C, H, W) format
    axes[0].set_title(image_title)

    # Plot the CAM mask
    axes[1].imshow(cam_mask_np[0], cmap='jet')  # Assuming your mask is grayscale
    axes[1].set_title('CAM Mask')

    # Plot the overlay (normalized)
    axes[2].imshow(masked_image_norm.transpose(1, 2, 0))  # Assuming your original image is in (C, H, W) format
    axes[2].set_title('Overlay (Normalized)')

    return fig

def showcam_withmask(img_tensor: torch.Tensor,
                     mask_tensor: torch.Tensor,
                     use_rgb: bool = False,
                     colormap: int = cv2.COLORMAP_JET,
                     image_weight: float = 0.5,
                     image_title: str = 'Original Image') -> plt.Figure:
    """ This function overlays the CAM mask on the image as a heatmap and returns the Figure object.
    By default, the heatmap is in BGR format.

    :param img_tensor: The base image tensor in PyTorch format.
    :param mask_tensor: The CAM mask tensor in PyTorch format.
    :param use_rgb: Whether to use an RGB or BGR heatmap; set to True if 'img_tensor' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.

    :return: Matplotlib Figure object.
    """
    # Convert PyTorch tensors to NumPy arrays
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mask = mask_tensor.cpu().numpy()

    # Convert the mask to a single-channel image
    mask_single_channel = np.uint8(255 * mask[0])

    heatmap = cv2.applyColorMap(mask_single_channel, colormap)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should be in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)

    # Create Matplotlib Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(img)
    axes[0].set_title(image_title)

    # Plot the CAM mask
    axes[1].imshow(mask[0], cmap='jet')
    axes[1].set_title('CAM Mask')

    # Plot the overlay
    axes[2].imshow(cam)
    axes[2].set_title('Overlay')

    return fig

def predict_and_gradcam(pil_image, model, target=100, plot_type='withmask'):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        split_white_and_gray(120),
    ])
    white_matter_tensor, gray_matter_tensor, origin_tensor = transform(pil_image)
    white_matter_tensor, gray_matter_tensor, origin_tensor = white_matter_tensor.unsqueeze(0).to(torch.float32),\
        gray_matter_tensor.unsqueeze(0).to(torch.float32),\
        origin_tensor.unsqueeze(0).to(torch.float32)
    
    def calculate_gradcammask(model_grad, input_tensor):
        target_layer = [model_grad.layer4[-1]] 
        gradcam = GradCAM(model=model_grad, target_layers=target_layer)
        targets = [ClassifierOutputTarget(target)]
        grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        grayscale_cam = torch.tensor(grayscale_cam)

        return grayscale_cam
    
    origin_model = model.resnet18_model
    white_model = model.whitematter_resnet18_model
    gray_model = model.graymatter_resnet18_model

    origin_cam = calculate_gradcammask(origin_model, origin_tensor)
    white_cam = calculate_gradcammask(white_model, white_matter_tensor)
    gray_cam = calculate_gradcammask(gray_model, gray_matter_tensor)

    class_idx = {0: 'Moderate Demented', 1: 'Mild Demented', 2: 'Very Mild Demented', 3: 'Non Demented'}
    prediction = model(white_matter_tensor, gray_matter_tensor, origin_tensor)
    predicted_class_index = torch.argmax(prediction).item()
    predicted_class_label = class_idx[predicted_class_index]

    if plot_type == 'withmask':
        return  predicted_class_label, showcam_withmask(torch.squeeze(origin_tensor), origin_cam),\
                showcam_withmask(torch.squeeze(white_matter_tensor), white_cam, image_title='White Matter'),\
                showcam_withmask(torch.squeeze(gray_matter_tensor), gray_cam, image_title='Gray Matter')
    elif plot_type == 'withoutmask':
        return  predicted_class_label, showcam_withoutmask(torch.squeeze(origin_tensor),origin_cam),\
                showcam_withoutmask(torch.squeeze(white_matter_tensor),white_cam, image_title='White Matter'),\
                showcam_withoutmask(torch.squeeze(gray_matter_tensor),gray_cam , image_title='Gray Matter')
    else:
        raise ValueError("plot_type must be either 'withmask' or 'withoutmask'")
