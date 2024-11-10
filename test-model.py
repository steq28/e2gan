import torch
from PIL import Image
from generator import Generator
from torchvision import transforms
import matplotlib.pyplot as plt

def load_and_test_model(checkpoint_path, test_image_path, output_path='output.png'):
    # Configurazione device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Trasformazioni per l'immagine di input
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Trasformazione inversa per visualizzare l'output
    inverse_transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage()
    ])
    
    # Carica il modello
    generator = Generator()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    # Carica e preprocessa l'immagine di test
    test_image = Image.open(test_image_path).convert('RGB')
    original_size = test_image.size
    
    # Prima converti in tensor, poi aggiungi la dimensione del batch
    input_tensor = transform(test_image)  # Questo è già un tensor
    input_tensor = input_tensor.unsqueeze(0)  # Aggiungi dimensione batch
    input_tensor = input_tensor.to(device)
    
    # Genera l'output
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    
    # Converti l'output in immagine
    output_image = inverse_transform(output_tensor.squeeze(0).cpu())
    
    # Ridimensiona all'originale se necessario
    output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
    
    # Salva l'output
    output_image.save(output_path)
    
    # Visualizza input e output
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Generated Image')
    plt.axis('off')
    
    plt.show()
    
    return output_image

if __name__ == "__main__":
    checkpoint_path = 'e2gan/checkpoint_epoch_100.pt'  # Sostituisci con il tuo percorso
    test_image_path = 'e2gan/obama.jpg'          # Sostituisci con il tuo percorso
    
    output_image = load_and_test_model(
        checkpoint_path=checkpoint_path,
        test_image_path=test_image_path,
        output_path='generated_output.png'
    )