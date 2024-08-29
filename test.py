import os
import json
import numpy as np
import cv2
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def load_embeddings(weights_root):
    embedding_db = {}
    embedding_files = sorted(os.listdir(weights_root))
    
    for idx, file_name in enumerate(embedding_files):
        if file_name.endswith('.json'):
            with open(os.path.join(weights_root, file_name), 'r') as f:
                embedding_data = json.load(f)
                embedding_db[file_name] = embedding_data
    
    return embedding_db, embedding_files

def identify_character(image_path, face_app, embedding_db, selected_models=None):
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # Convert BGR to RGB
    cv2_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Get the face embeddings
    faces = face_app.get(cv2_image)
    if not faces:
        raise ValueError("No faces detected in the image.")

    embedding = faces[0].normed_embedding

    # Initialize variables to track the best match
    max_similarity = -1
    identified_character = None
    results = []

    # Compare against each selected model
    for file_name, embedding_data in embedding_db.items():
        if selected_models and file_name not in selected_models:
            continue
        for character_name, character_embedding in embedding_data.items():
            similarity = cosine_similarity(embedding, np.array(character_embedding))
            results.append((character_name, similarity))
            if similarity > max_similarity:
                max_similarity = similarity
                identified_character = character_name

    # Print all comparisons
    print("Comparisons:")
    for character_name, similarity in results:
        print(f"{character_name}: {similarity}")

    return identified_character, max_similarity

def main():
    # Download the model files if not already present
    snapshot_download(
        repo_id="fal/AuraFace-v1",
        local_dir="models/auraface",
    )

    # Initialize the FaceAnalysis model
    face_app = FaceAnalysis(name="auraface", providers=["CPUExecutionProvider"], root=".")
    face_app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU, adjust for GPU if available

    # Define paths
    inference_root = "./test-input"
    weights_root = "./weights"
    
    # Load the embedding database and list available models
    embedding_db, embedding_files = load_embeddings(weights_root)

    if not embedding_db:
        raise ValueError(f"No embeddings found in {weights_root}.")

    # List all available models
    print("Available models:")
    for idx, file_name in enumerate(embedding_files, start=1):
        print(f"{idx}. {file_name}")

    # User input for model selection
    selected = input("Enter the model numbers to compare against (comma separated), or 0 for all models: ").strip()

    if selected == "0":
        selected_models = None  # Compare against all models
    else:
        selected_indices = [int(i) for i in selected.split(',')]
        selected_models = [embedding_files[i-1] for i in selected_indices]

    # Loop through all images in the inference directory
    for test_image_name in os.listdir(inference_root):
        test_image_path = os.path.join(inference_root, test_image_name)
        if not test_image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip non-image files

        print(f"\nProcessing {test_image_name}...")

        try:
            # Identify the character in the test image
            identified_character, max_similarity = identify_character(test_image_path, face_app, embedding_db, selected_models)

            # Print the best match
            if identified_character:
                print(f"Best match: {identified_character} with similarity {max_similarity}.")
            else:
                print("No match found.")
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main()
