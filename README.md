# Aura Face Detect
## What is this?
This is a simple, light-weight, extremely fast face detection tool created by [fal](https://huggingface.co/fal) and released at [fal/AuraFace-v1 on huggingface](https://huggingface.co/fal/AuraFace-v1).

My contribution here is a quick and dirty "training" tool, and a "test" tool.

train.py lets you create an embedding for one or more input images of a character's face, and save this embedding to a file.

test.py lets you compare a number of images in one folder, against one or more of the saved embeddings, returning a value of how close the face matches. Basically a face similarity detection score from 0 to 1.

## Setup
1. Run `venv_create.bat` or create your own virtual environment.
2. Install the requirements from `requirements.txt`. (`pip install -r requirements.txt`). Skip this step if you installed the requirements via the `venv_create.bat` setup file.

## How to "train"
Run `py train.py` from the (venv), to create embeddings. One embedding will be created per subfolder in `/train-input/`.

## How to "test"
Run `py test.py` from the (venv), to compare each image in `/test-input/` against the embeddings you have. You will be asked which embedding to compare the images against. Press ENTER or 0 to compare against all, or choose the embedding with numbers as instructed.
