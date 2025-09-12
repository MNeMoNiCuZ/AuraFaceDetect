import os
import json
import numpy as np
import cv2
from datetime import datetime
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
import shutil
import argparse

# --- Configuration ---
REPO_ID = "fal/AuraFace-v1"
MODEL_DIR = "models/auraface"
DET_SIZE = (640, 640)
INFERENCE_ROOT = "./test-input"
WEIGHTS_ROOT = "./embeddings"
OUTPUT_ROOT = "./output"
DEFAULT_SNAP_PERCENTAGE = 1
DEFAULT_MODEL_SELECTION = "1"
DEFAULT_ACTION = "2"  # 1 for Print, 2 for Sort
SORT_ACTION = "copy" # 'move' or 'copy'
# ---------------------

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def load_embeddings(weights_root):
    embedding_db = {}
    
    # Get json files and sort by creation date
    json_files = [f for f in os.listdir(weights_root) if f.endswith('.json')]
    # Create a list of (file_path, creation_time) tuples
    files_with_time = [(f, os.path.getctime(os.path.join(weights_root, f))) for f in json_files]
    # Sort the list by creation_time (the second element of the tuple)
    files_with_time.sort(key=lambda x: x[1])
    # Get just the filenames in the new sorted order
    embedding_files = [f[0] for f in files_with_time]

    for file_name in embedding_files:
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
    print("\nComparisons:")
    for character_name, similarity in results:
        print(f"{character_name}: {similarity}")

    return identified_character, max_similarity

def main(args):
    # Download the model files if not already present
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
    )

    # Initialize the FaceAnalysis model
    face_app = FaceAnalysis(name="auraface", providers=["CPUExecutionProvider"], root=".")
    face_app.prepare(ctx_id=-1, det_size=DET_SIZE)

    # Define paths from args
    inference_root = args.inference_root
    weights_root = args.weights_root
    
    # Load the embedding database and list available models
    embedding_db, embedding_files = load_embeddings(weights_root)

    if not embedding_db:
        raise ValueError(f"No embeddings found in {weights_root}.")

    # --- Model Selection ---
    selected = args.model_selection
    if selected is None: # If no CLI arg, go interactive
        print("\n--- Available Models (sorted by creation date) ---")
        print("="*70)
        print(f"{ 'No.':<5}{'Model Name':<35}{'Creation Date':<30}")
        print("-"*70)
        for idx, file_name in enumerate(embedding_files, start=1):
            file_path = os.path.join(weights_root, file_name)
            try:
                creation_time = os.path.getctime(file_path)
                creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            except FileNotFoundError:
                creation_date = "N/A"
            print(f"{idx:<5}{file_name:<35}{creation_date:<30}")
        print("="*70)
        selected = input(f"Enter model numbers (comma separated), or press ENTER for default ({DEFAULT_MODEL_SELECTION}): ").strip() or DEFAULT_MODEL_SELECTION

    if selected.upper() == "ALL":
        selected_models = None # Compare against all models
    else:
        try:
            selected_indices = [int(i) for i in selected.split(',')]
            selected_models = [embedding_files[i-1] for i in selected_indices]
        except (ValueError, IndexError):
            print("Invalid selection. Please enter numbers from the list.")
            return
    # -----------------------

    # --- Action Selection ---
    action_choice = args.action
    if action_choice is None: # If no CLI arg, go interactive
        print("\n--- Choose Action ---")
        print("1. Print results to console")
        print("2. Sort images into folders by similarity")
        action_choice = input(f"Select an action (press ENTER for default: {DEFAULT_ACTION}): ").strip() or DEFAULT_ACTION

    snap_percentage = args.snap_percentage
    if action_choice == '2':
        if args.action is None: # action was chosen interactively
            print("\n--- Configure Sorting ---")
            snap_input = input(f"Enter the % 'snap' for sorting (1, 5, or 10, default is {DEFAULT_SNAP_PERCENTAGE}): ").strip()
            if snap_input in ('1', '5', '10'):
                snap_percentage = int(snap_input)
        
        sorted_output_root = args.output_root
        if not os.path.exists(sorted_output_root):
            os.makedirs(sorted_output_root)
        print("-" * 25)
    # ------------------------

    print("\n--- Processing Images ---")
    # Get a list of files to process to avoid issues with modifying the directory while iterating
    image_files_to_process = [f for f in os.listdir(inference_root) if not os.path.isdir(os.path.join(inference_root, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for test_image_name in image_files_to_process:
        test_image_path = os.path.join(inference_root, test_image_name)
        
        print(f"\nProcessing {test_image_name}...")

        try:
            identified_character, max_similarity = identify_character(test_image_path, face_app, embedding_db, selected_models)

            if action_choice == '2': # Corresponds to "Sort"
                if identified_character:
                    similarity_percent = max_similarity * 100
                    snap_folder_start = int(similarity_percent / snap_percentage) * snap_percentage
                    folder_name = str(snap_folder_start)
                    
                    target_dir = os.path.join(args.output_root, folder_name)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    
                    new_image_path = os.path.join(target_dir, test_image_name)
                    if args.sort_action == 'copy':
                        shutil.copy(test_image_path, new_image_path)
                        print(f"-> Copied '{test_image_name}' to '{target_dir}' with similarity {max_similarity:.2%}")
                    else: # default to move
                        shutil.move(test_image_path, new_image_path)
                        print(f"-> Moved '{test_image_name}' to '{target_dir}' with similarity {max_similarity:.2%}")
                else:
                    print("-> No match found, image not moved.")
            else: # Default to printing
                if identified_character:
                    print(f"-> Best match: {identified_character} with similarity {max_similarity:.2%}.")
                else:
                    print("-> No match found.")

        except ValueError as e:
            print(e)
    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify faces in images and sort them.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_selection', type=str, default=None, help='Model numbers to use, comma-separated (e.g., "1,2") or "ALL". If not provided, runs in interactive mode.')
    parser.add_argument('--action', type=str, default=None, choices=['1', '2'], help='Action to perform: 1 for Print, 2 for Sort. If not provided, runs in interactive mode.')
    parser.add_argument('--snap_percentage', type=int, default=DEFAULT_SNAP_PERCENTAGE, choices=[1, 5, 10], help='Snap percentage for sorting.')
    parser.add_argument('--sort_action', type=str, default=SORT_ACTION, choices=['move', 'copy'], help="Action for sorting files: 'move' or 'copy'.")
    parser.add_argument('--inference_root', type=str, default=INFERENCE_ROOT, help='Directory with images to process.')
    parser.add_argument('--weights_root', type=str, default=WEIGHTS_ROOT, help='Directory with embedding files.')
    parser.add_argument('--output_root', type=str, default=OUTPUT_ROOT, help='Directory for sorted output.')
    
    args = parser.parse_args()
    main(args)
