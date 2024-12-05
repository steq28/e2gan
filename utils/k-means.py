!pip install transformers shutil
import os
import torch
import zipfile
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime

class ImageClusterSelector:
    def __init__(self, n_clusters, input_dir, output_dir, device=None):
        """
        Initialize the image cluster selector.
        
        Args:
            n_clusters (int): Number of clusters for K-means
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save selected images
            device (str, optional): Device to use for computation
        """
        self.n_clusters = n_clusters
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def extract_embedding(self, image_path):
        """Extract embedding from a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(image)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            return None

    def get_image_paths(self):
        """Get valid image paths from input directory."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        image_paths = [
            os.path.join(self.input_dir, img) 
            for img in os.listdir(self.input_dir) 
            if img.lower().endswith(valid_extensions)
        ]
        if not image_paths:
            raise ValueError(f"No valid images found in {self.input_dir}")
        return image_paths

    def create_zip_output(self, selected_images, zip_filename=None):
        """
        Create a zip file containing the selected images.
        
        Args:
            selected_images (list): List of paths to selected images
            zip_filename (str, optional): Name of the zip file. If None, generates a timestamp-based name
        """
        if zip_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = os.path.join(self.output_dir, f'selected_images_{timestamp}.zip')
        else:
            zip_filename = os.path.join(self.output_dir, zip_filename)

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_path in tqdm(selected_images, desc="Adding images to zip"):
                image_name = os.path.basename(img_path)
                zipf.write(img_path, image_name)
        
        return zip_filename

    def process_images(self, create_zip=True, zip_filename=None):
        """Main processing pipeline."""
        try:
            # Get image paths
            image_paths = self.get_image_paths()

            # Extract embeddings
            embeddings_list = []
            valid_paths = []
            
            for img_path in tqdm(image_paths, desc="Processing images"):
                embedding = self.extract_embedding(img_path)
                if embedding is not None:
                    embeddings_list.append(embedding)
                    valid_paths.append(img_path)
            
            embeddings = np.array(embeddings_list)
            
            # Adjust n_clusters if necessary
            self.n_clusters = min(self.n_clusters, len(valid_paths))

            # Perform clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            kmeans.fit(embeddings)

            selected_images = []
            for cluster_id in range(self.n_clusters):
                cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
                cluster_embeddings = embeddings[cluster_indices]
                distances = np.linalg.norm(
                    cluster_embeddings - kmeans.cluster_centers_[cluster_id],
                    axis=1
                )
                closest_index = cluster_indices[np.argmin(distances)]
                selected_images.append(valid_paths[closest_index])

            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            if create_zip:
                # Create zip file instead of copying files
                zip_path = self.create_zip_output(selected_images, zip_filename)
            else:
                # Copy selected images as before
                for img_path in selected_images:
                    image_name = os.path.basename(img_path)
                    output_path = os.path.join(self.output_dir, image_name)
                    shutil.copy2(img_path, output_path)
            return selected_images

        except Exception as e:
            raise

def process_all_folders_in_directory(parent_dir, n_clusters, output_dir):
    """
    Process all subfolders inside a parent directory, creating a zip file for each.

    Args:
        parent_dir (str): Parent directory containing folders to process
        n_clusters (int): Number of clusters for image selection
        output_dir (str): Directory to save output zip files
    """
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    
    for folder in folders:
        input_dir = os.path.join(parent_dir, folder)
        folder_output_dir = os.path.join(output_dir, folder)
        os.makedirs(folder_output_dir, exist_ok=True)

        print(f"Processing folder: {folder}")
        selector = ImageClusterSelector(
            n_clusters=n_clusters,
            input_dir=input_dir,
            output_dir=folder_output_dir
        )
        selector.process_images(create_zip=True, zip_filename=f"{folder}_selected_images.zip")

# Main script
if __name__ == "__main__":
    parent_directory = "/kaggle/input/modified_images"  # Parent directory containing subfolders
    output_directory = "output_zips"  # Directory to save the zip files
    n_clusters = 300  # Number of clusters for image selection

    os.makedirs(output_directory, exist_ok=True)
    process_all_folders_in_directory(parent_directory, n_clusters, output_directory)
