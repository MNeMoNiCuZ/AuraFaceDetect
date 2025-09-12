# Aura Face Detect
## What is this?
This is a simple, light-weight, extremely fast face detection tool created by [fal](https://huggingface.co/fal) and released at [fal/AuraFace-v1 on huggingface](https://huggingface.co/fal/AuraFace-v1).

My contribution here is a quick and dirty "training" tool, and a "test" tool.

train.py lets you create an embedding for one or more input images of a character's face, and save this embedding to a file.

test.py lets you compare a number of images in one folder, against one or more of the saved embeddings, returning a value of how close the face matches. Basically a face similarity detection score from 0 to 1.

## Setup
1. Run `venv_create.bat` or create your own virtual environment.
2. Install the requirements from `requirements.txt`. (`pip install -r requirements.txt`). Skip this step if you installed the requirements via the `venv_create.bat` setup file.

![image](https://github.com/user-attachments/assets/59e239c7-947a-42ef-80ee-67e1b2faecd9)


## How to "train"
Run `py train.py` from the (venv), to create embeddings.

One embedding will be created per subfolder in `/train-input/`.

To "train" multiple characters at the same time, create one folder for each character and put the face-images of each character in their respective folder.
![image](https://github.com/user-attachments/assets/52a23016-826f-4bda-bf26-32c806195d07)

## How to "test"
Run `py test.py` from the (venv), to compare each image in `/test-input/` against the embeddings you have.

You will be asked which embedding to compare the images against.

Press ENTER to choose the default one, or 0 to compare against all, or choose the embedding with numbers as instructed.

You will then be asked which action to perform:

1. Print results to console
2. Sort images into folders by similarity

If you choose #2, you are then asked the snap % for the sorting. Use this to create ranges (sort by every 5% of similarity) for example.

<img width="1046" height="1128" alt="image" src="https://github.com/user-attachments/assets/a580de46-e217-4680-aa4a-4e52fb2b4599" />
