from datasets import load_dataset

dataset = load_dataset('csv', data_dir='Datasets', cache_dir='Datasets/cache')

print(dataset)