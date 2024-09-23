import torch
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from model_download import download_model

from config import save_directory
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup, T5Tokenizer
import pytorch_lightning as pl

tokenizer = T5Tokenizer.from_pretrained(save_directory)

prefix = "generate api: "
max_input_length = 256
max_target_length = 256

def preprocess_examples(examples):

  apis = examples['input']
  hidden_apis = examples['output']

  inputs = [prefix + api for api in apis]
  model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

  labels = tokenizer(hidden_apis, max_length=max_target_length, padding="max_length", truncation=True).input_ids

  # 由于padding会填充为0，这里需要将labels中出现的0改为-100方便训练
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)

  # model_inputs本身就有input_ids, attention_mask两列了，现在再加一列
  model_inputs["labels"] = labels_with_ignore_index

  return model_inputs

class APISeer(pl.LightningModule):
    # 学习速率，回合数，热身步数
    def __init__(self, lr=5e-5, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(save_directory)
        # 将超参数保存到模型，方便以后使用
        self.save_hyperparameters()

    # 前向传播得到输出，下面的self()即使用forward函数
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    # 下面的三个步骤统一使用该函数，返回loss
    def common_step(self, batch, batch_idx):
        # 将batch按键值解包为参数
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # 使用AdamW优化器
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # 使用 learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader

if __name__=='__main__':
    download_model()

    dataset = load_dataset('csv', data_dir='Datasets/9-13')

    dataset = dataset.map(preprocess_examples, batched=True)
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=8)
    valid_dataloader = DataLoader(dataset['validation'], batch_size=4)
    test_dataloader = DataLoader(dataset['test'], batch_size=4)

    print("Finish generating datasets.")


    model = APISeer()
    torch.set_float32_matmul_precision('high')

    wandb_logger = WandbLogger(name='APISeer-9-13-base', project='APISeer')

    # 根据validation_loss提早结束train过程
    # 3回合loss变化范围不大即可停止
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(devices=1, accelerator='gpu', logger=wandb_logger, callbacks=[early_stop_callback, lr_monitor])
    trainer.fit(model)

    save_trained = "Trained/APISeer_Trained-9-13-base"
    model.model.save_pretrained(save_trained)


