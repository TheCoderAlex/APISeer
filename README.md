## APISeer

> 基于T5模型的大语言隐藏API生成模型 </br>
> Author: Zifeng Tang

## 使用

1. 首先建议在虚拟环境中安装所需要的依赖，其中 `Pytorch` 依赖可能需要根据您所使用的操作系统类别和版本进行安装。

```shell
pip install -r requirements.txt
```

2. 在使用前在 `config.py` 文件中定义好想选用的模型，以及训练好的模型所存放的位置。以下是一种使用样例。

```python
# Model Config
model_name = 't5-base' # or "t5-small", "t5-large", "t5-3b", "t5-11b
save_directory = f"./Model/{model_name}"
```

3. 将处理好的数据集放置在 `Datasets` 文件夹下，并在 `train.py` 的 `108` 行进行修改。 可以使用提供的 `DataProcess` 工具进行 `json` 至 `csv` 文件的转换。

4. 运行 `train.py` 程序。

```shell
python train.py
```

注意，在第一次运行时，程序会自动下载相应的模型文件和对应模型的 `Tokenizer`，之后不会重复下载，除非修改模型。

模型训练默认使用GPU，如有修改请跳转至 `train.py` 的 `137` 行自行修改。

模型训练使用 `W&B` Logger 进行数据记录与分析，第一次运行需要登录相应的账号。

5. 测试模型

在 `test.py` 中更换 `model` 和 `text` 变量的值，运行 `test.py` 即可得到测试结果。

```shell
python test.py
```

## 效果

参考作品报告。