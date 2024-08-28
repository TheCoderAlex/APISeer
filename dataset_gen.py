from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from config import save_directory

tokenizer = T5Tokenizer.from_pretrained(save_directory)

prefix = "Summarize Python: "
max_input_length = 256
max_target_length = 128

def preprocess_examples(examples):
  # examples参数实际上是一个batch
  codes = examples['code']
  docstrings = examples['docstring']

  # 将输入的多行code整合成一行，使用tokenizer得到张量
  inputs = [prefix + code for code in codes]
  model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

  # 将输出编码
  labels = tokenizer(docstrings, max_length=max_target_length, padding="max_length", truncation=True).input_ids

  # 由于padding会填充为0，这里需要将labels中出现的0改为-100方便训练
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)

  # model_inputs本身就有input_ids, attention_mask两列了，现在再加一列
  model_inputs["labels"] = labels_with_ignore_index

  return model_inputs

dataset = load_dataset('csv', data_dir='Datasets', cache_dir='Datasets/cache')
dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])


train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=8, num_workers=4, persistent_workers=True)
valid_dataloader = DataLoader(dataset['validation'], batch_size=4, num_workers=4, persistent_workers=True)
test_dataloader = DataLoader(dataset['test'], batch_size=4, num_workers=4, persistent_workers=True)