import os
import json
import numpy as np
import cv2
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis

# Configuration
overwrite = True  # Set to True to overwrite existing models

def get_image_embeddings(folder_path, face_app):
    embeddings = []
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file_name)
            input_image = cv2.imread(img_path)
            if input_image is None:
                continue

            # Convert BGR to RGB
            cv2_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            # Get the face embeddings
            faces = face_app.get(cv2_image)
            if faces:
                embeddings.append(faces[0].normed_embedding)

    if not embeddings:
        raise ValueError(f"No valid face embeddings found in folder: {folder_path}")
    
    # Average all embeddings to create a single embedding for the character
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def main():
    # Download the model files if not already present
    snapshot_download(
        repo_id="fal/AuraFace-v1",
        local_dir="models/auraface",
    )

    # Initialize the FaceAnalysis model
    face_app = FaceAnalysis(name="auraface", providers=["CPUExecutionProvider"], root=".")
    face_app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU, adjust for GPU if available

    # Define input and output paths
    input_root = "./train-input"
    output_root = "./weights"

    # Ensure output directory exists
    os.makedirs(output_root, exist_ok=True)

    # Process each character subfolder in /input/
    for character_name in os.listdir(input_root):
        character_folder = os.path.join(input_root, character_name)
        if not os.path.isdir(character_folder):
            continue

        output_file = os.path.join(output_root, f"{character_name}.json")

        # Skip if output file exists and overwrite is False
        if os.path.exists(output_file) and not overwrite:
            print(f"Model for {character_name} already exists. Skipping...")
            continue

        # Get the average embedding for the character
        try:
            avg_embedding = get_image_embeddings(character_folder, face_app)
        except ValueError as e:
            print(e)
            continue

        # Save the character's embedding in the output folder
        with open(output_file, 'w') as f:
            json.dump({character_name: avg_embedding.tolist()}, f)

        print(f"Embedding for {character_name} saved successfully.")

if __name__ == "__main__":
    main()
