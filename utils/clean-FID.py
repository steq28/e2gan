!pip install clean-fid
import os
import zipfile
from cleanfid import fid
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


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

    # plt.imshow(generated_image)
    # plt.show()

    return original_image, generated_image

# Prepara le trasformazioni
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

start_index = 1001

prompts = ["convert to albino style", "make the hair blonde",
           "apply van gogh style", "make the person old",
           "make the hair pink"]

style = ['albino', 'blonde', 'gogh', 'old', 'pink']

# Loop through each style and process images
for j in range(len(prompts)):
    img_index = 1001
    for i in range(30):
        original_path = f'/kaggle/input/test-multiprompt/original_images/{start_index+i}.png'

        destination_path = f'/kaggle/working/{style[j]}/{img_index + i}.png'
        _, generated_image = process_single_image(original_path, generator, prompts[j], transform, device)
        
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Save the generated image
        generated_image.save(destination_path)


    # Compress each style's folder into a zip file for download
    zip_file_path = f'/kaggle/working/{style[j]}.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for root, dirs, files in os.walk(f'/kaggle/working/{style[j]}'):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=os.path.join(style[j], file))

# Compute FID scores
fid_scores = []

for s in style:
    diff_path = f'/kaggle/input/test-multiprompt/{s}'
    gen_path = f'/kaggle/working/{s}'
    fid_scores.append(fid.compute_fid(diff_path, gen_path))

# Calculate the median FID score
median_fid = np.mean(fid_scores)
print(f"The overall FID score is {median_fid}")
print(fid_scores)
