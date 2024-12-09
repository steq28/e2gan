!pip install clean-fid
import os
import zipfile
from cleanfid import fid
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import shutil


def generate_image(original_tensor, generator, text_prompt, device):
    """Genera un'immagine utilizzando il tensore originale e il prompt."""
    original_tensor = original_tensor.unsqueeze(0).to(device)  # Aggiungi dimensione batch
    generator.eval()
    with torch.no_grad():
        generated_tensor = generator(original_tensor, text_prompt)
    created = torch.clamp((generated_tensor.squeeze(0).cpu() + 1) / 2, 0, 1)  # Denormalizzazione

    created = created.permute(1, 2, 0).numpy()  # Da (C, H, W) a (H, W, C)

    #created = to_pil_image(generated_tensor.squeeze(0).cpu())

    # plt.imshow(created)
    # plt.show()

    created = to_pil_image(created)
    
    return created


prompts = ["convert to albino style", "make the hair blonde",
           "apply van gogh style", "make the person old",
           "make the hair pink"]

styles = ['albino', 'blonde', 'gogh', 'old', 'pink']


test0, test1,test2, test3, test4 = [], [], [], [], []
# test1 = []
# test2 = []
# test3 = []
# test4 = []

for source_image, target_image, prompt in test_dataset:
    if prompt == prompts[0]:
        if len(test0) < 30:
            test0.append((source_image, target_image, prompt))
    elif prompt == prompts[1]:
        if len(test1) < 30:
            test1.append((source_image, target_image, prompt))
    elif prompt == prompts[2]:
        if len(test2) < 30:
            test2.append((source_image, target_image, prompt))
    elif prompt == prompts[3]:
        if len(test3) < 30:
            test3.append((source_image, target_image, prompt))
    elif prompt == prompts[4]:
        if len(test4) < 30:
            test4.append((source_image, target_image, prompt))

print(len(test0),len(test1),len(test2),len(test3),len(test4))

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

test_lists = [test0, test1, test2, test3, test4]

# Generate image from triple (original, modified, prompt)
start_index = 1001
for style_idx, prompt in enumerate(prompts):
    style_folder = styles[style_idx]
    os.makedirs(f'/kaggle/working/{style_folder}', exist_ok=True)

    for i, (original_tensor, _, _) in enumerate(test_lists[style_idx]):

        # Generate image
        generated_image = generate_image(original_tensor, generator, prompt, device)

        # Save
        save_path = f'/kaggle/working/{style_folder}/{start_index + i}.png'
        generated_image.save(save_path)
        #print(f"Saved generated image to: {save_path}")

    # Compress
    zip_file_path = f'/kaggle/working/{style_folder}.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for root, _, files in os.walk(f'/kaggle/working/{style_folder}'):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=os.path.join(style_folder, file))
    #print(f"Compressed images to: {zip_file_path}")


fid_scores = []

# Calcolo FID utilizzando immagini dalle triple
for style_idx, style_folder in enumerate(styles):
    diff_dir = f'/kaggle/working/diff_{style_folder}'

    os.makedirs(diff_dir, exist_ok=True)

    # Salva le immagini reali (modified) e generate in directory temporanee
    for i, (_, modified_tensor, _) in enumerate(test_lists[style_idx]):

        # Salva immagine modificata (real)
        diff_image_path = os.path.join(diff_dir, f"diff_{style_folder}_{i}.png")
        diff_image = to_pil_image(torch.clamp((modified_tensor + 1) / 2, 0, 1))  # Denormalizzazione
        diff_image.save(diff_image_path)

    # Calcola FID usando cleanfid
    score = fid.compute_fid(diff_dir, f'/kaggle/working/{style_folder}')
    fid_scores.append(score)
    print(f"FID for {style_folder}: {score}")

    # Pulisce le directory temporanee
    shutil.rmtree(diff_dir)

# Calcola FID medio
median_fid = np.median(fid_scores)
print(f"The overall FID score is {median_fid}")
print(fid_scores)
