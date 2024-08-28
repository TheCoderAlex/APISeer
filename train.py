from transformers import T5Tokenizer

from model_download import download_model
from config import save_directory

download_model()
tokenizer = T5Tokenizer.from_pretrained(save_directory)

input = 'Hello'
out = tokenizer(input, max_length=128, padding="max_length", truncation=True)

print(out)