import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def load_model(generator, discriminator, checkpoint_path, device):
    """Carica i pesi del modello dal file di checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    print(f"Modello caricato dal checkpoint: {checkpoint_path}")

def process_single_image(image_path, generator, text_prompt, transform, device):
    """Processa un'immagine singola con il generatore e il prompt testuale."""
    # Caricare l'immagine
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # Applicare le trasformazioni
    image_tensor = transform(image).unsqueeze(0).to(device)  # Aggiungi dimensione batch
    
    # Generare l'immagine
    generator.eval()
    with torch.no_grad():
        generated_image_tensor = generator(image_tensor, text_prompt)
    
    # Convertire in immagine PIL
    generated_image_tensor = generated_image_tensor.squeeze(0).cpu()
    generated_image = ToPILImage()(torch.clamp((generated_image_tensor + 1) / 2, 0, 1))  # Denormalizzazione
    
    return original_image, generated_image

def visualize_images(original_image, generated_image):
    """Mostra l'immagine originale e quella generata fianco a fianco."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].axis("off")
    axes[0].set_title("Input")
    
    axes[1].imshow(generated_image)
    axes[1].axis("off")
    axes[1].set_title("Output")
    
    plt.show()

# Specifica il percorso dell'immagine e il prompt
image_path = '/kaggle/input/obamaa/obama.jpg'
#checkpoint_path = '/kaggle/working/models/final_model.pth'
checkpoint_path = '/kaggle/input/e2gan-v6/pytorch/default/2/e2gan-v6-2'
text_prompt = "albino"  # Cambia il prompt se necessario

# Prepara le trasformazioni
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
    
# Inizializza i modelli
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = TextGuidedGenerator(device=device).to(device)
discriminator = Discriminator().to(device)
    
# Caricare il modello addestrato
load_model(generator, discriminator, checkpoint_path, device)
    
# Generare immagine
import time
start_time = time.time()
original_image, generated_image = process_single_image(image_path, generator, text_prompt, transform, device)
end_time = time.time() - start_time

print(end_time*1000)
    
# Visualizzare il risultato
visualize_images(original_image, generated_image)
