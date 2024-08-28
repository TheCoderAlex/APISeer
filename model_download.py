import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import model_name, save_directory


def is_directory_empty(directory):
    return not any(os.scandir(directory))


def download_model():
    if not os.path.exists(save_directory) or is_directory_empty(save_directory):
        print(f"Model '{model_name}' does not exist. Downloading the model...")

        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

        print(f"Model and tokenizer downloaded and saved to {save_directory}")
    else:
        print(f"Model and tokenizer already exist in {save_directory}, skipping download.")
