import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

def load_model(model_path, device):
    """Load the saved model with DataParallel handling."""
    checkpoint = torch.load(model_path)
    # Initialize the model with the correct number of styles (3 in this case)
    model = ImprovedGenerator(num_styles=3).to(device)
    
    # Remove the 'module.' prefix from state_dict keys
    state_dict = checkpoint['generator']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    """Preprocess input image."""
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    return transform(image).unsqueeze(0)

def test_model(model, input_image, style_idx, device):
    """Run model inference with a specified style index."""
    with torch.no_grad():
        input_tensor = input_image.to(device)
        style_idx_tensor = torch.tensor([style_idx], device=device)
        output = model(input_tensor, style_idx_tensor)
    return output

def visualize_results(original, generated, style_names=['Albino', 'Blonde', 'Van Gogh'], style_idx=0):
    """Visualize original and generated images."""
    # Denormalize images
    original = original * 0.5 + 0.5
    generated = generated * 0.5 + 0.5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(transforms.ToPILImage()(original.squeeze(0).cpu()))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(transforms.ToPILImage()(generated.squeeze(0).cpu()))
    ax2.set_title(f'Generated Image ({style_names[style_idx]} Style)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/kaggle/working/models/best_model.pth'
    input_image_path = '/kaggle/input/atml-imagesv8/obama.jpg'
    
    # Style index mapping:
    # 0: Albino
    # 1: Blonde
    # 2: Van Gogh
    style_idx = 0  # Change this to test different styles
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load model
    model = load_model(model_path, device)
    
    # Preprocess input image
    input_image = preprocess_image(input_image_path, transform)
    
    # Generate output
    generated_image = test_model(model, input_image, style_idx, device)
    print(f"Input shape: {input_image.shape}, Output shape: {generated_image.shape}")
    
    # Save the image
    output_path = f"/kaggle/working/generated_image_style_{style_idx}.png"
    torchvision.utils.save_image(generated_image, output_path, normalize=True, format="PNG")
    print(f"Saved image in: {output_path}")
    
    # Visualize results
    style_names = ['Albino', 'Blonde', 'Van Gogh']
    visualize_results(input_image, generated_image, style_names, style_idx)

if __name__ == '__main__':
    main()
