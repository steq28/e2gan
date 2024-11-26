import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

# Funzione per caricare il modello
def load_model(model_path, generator_class, device):
    """Load the saved model with DataParallel handling"""
    checkpoint = torch.load(model_path)
    model = generator_class().to(device)
    
    # Remove the 'module.' prefix from state_dict keys
    state_dict = checkpoint['generator']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# Funzione per preprocessare l'immagine
def preprocess_image(image_path, transform):
    """Preprocess input image"""
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Funzione per eseguire l'inferenza sul modello
def test_model(model, input_image, device):
    """Run model inference"""
    with torch.no_grad():
        input_tensor = input_image.to(device)
        output = model(input_tensor)
    return output

# Funzione per visualizzare i risultati
def visualize_results(original, generated):
    """Visualize original and generated images"""
    # Denormalize images
    original = original * 0.5 + 0.5
    generated = generated * 0.5 + 0.5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(transforms.ToPILImage()(original.squeeze(0).cpu()))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(transforms.ToPILImage()(generated.squeeze(0).cpu()))
    ax2.set_title('Generated Image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Classe per il dataset personalizzato
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Funzione per fine-tuning sul concetto A
def fine_tune_on_concept_A(model, concept_A_images_folder, device, num_epochs=10):
    # Trasformazioni per le immagini
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Carica le immagini del concetto A
    concept_A_images = [os.path.join(concept_A_images_folder, f) for f in os.listdir(concept_A_images_folder) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"Loaded {len(concept_A_images)} images for concept A from {concept_A_images_folder}")
    
    # Verifica che ci siano immagini nel dataset
    if len(concept_A_images) == 0:
        print("Il dataset per il concetto A è vuoto. Verifica il percorso o le immagini.")
        return model

    # Crea il dataset per il concetto A
    dataset_A = CustomDataset(concept_A_images, transform=transform)
    
    # Crea il DataLoader
    dataloader_A = DataLoader(dataset_A, batch_size=8, shuffle=True)
    print(f"Numero di batch nel DataLoader: {len(dataloader_A)}")
    
    # Imposta l'ottimizzatore
    optimizer = Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader_A:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            
            # Usa la L1 loss per il fine-tuning (puoi cambiare la loss a seconda del task)
            loss = torch.nn.L1Loss()(output, batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
    return model

# Funzione principale
def main():
    # Configurazione
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/kaggle/working/models/best_model.pth'  # Path to saved model
    input_image_path = '/kaggle/input/atml-imagesv8/obama.jpg'  # Replace with actual path
    concept_A_images_folder = '/kaggle/input/multiprompt/gogh'  # Cartella contenente le immagini per il concetto A
    
    # Trasformazioni (dovrebbero essere le stesse usate durante l'addestramento)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Carica il modello
    model = load_model(model_path, ImprovedGenerator, device)
    
    # Fine-tuning sul concetto A
    model = fine_tune_on_concept_A(model, concept_A_images_folder, device)
    
    # Preprocessa l'immagine di input
    input_image = preprocess_image(input_image_path, transform)
    
    # Genera l'immagine
    generated_image = test_model(model, input_image, device)
    print(f"Input image shape: {input_image.shape}, Generated image shape: {generated_image.shape}")
    
    # Salva l'immagine generata
    output_path = "/kaggle/working/generated_image.png"  # Salva nella working directory di Kaggle
    torchvision.utils.save_image(generated_image, output_path, format="PNG")
    print(f"Immagine generata salvata in: {output_path}")
    
    # Visualizza i risultati
    visualize_results(input_image, generated_image)

# Esegui la funzione principale
if __name__ == '__main__':
    main()
