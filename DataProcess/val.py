import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('api_dataset.csv')

# 从数据中随机抽取5万条记录
sampled_df = df.sample(n=50000, random_state=42)

# 将抽取的记录保存到一个新的CSV文件中
sampled_df.to_csv('validation.csv', index=False)

# 从原始数据中删除抽取的5万条记录
remaining_df = df.drop(sampled_df.index)

# 将剩余的记录保存到原始文件或者另一个文件中
remaining_df.to_csv('train.csv', index=False)

