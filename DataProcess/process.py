import json
import csv


def generate_api_description(api_info):
    verb = api_info.get('verb', '').upper()
    request_path = api_info.get('requestPath', '')
    parameters = api_info.get('parameters', [])

    # 收集参数名称
    param_names = [param.get('name', '') for param in parameters]

    # 构建描述
    description = f"API uses {verb} method at `{request_path}`."
    if param_names:
        description += " Parameters: " + ", ".join(f"`{name}`" for name in param_names) + "."

    return description  # 不再截断描述


# 加载 JSON 文件中的 API 数据
with open('api_info_0913_updated_2.json', 'r', encoding='utf-8') as f:
    api_list = json.load(f)

# 创建一个从 API ID 到 API 数据的映射
api_id_map = {api['id']: api for api in api_list}

# 打开 CSV 文件以写入
with open('api_dataset.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['input', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    # 处理列表中的每个 API
    for api in api_list:
        # 生成可见 API 的输入描述
        input_description = generate_api_description(api)

        # 检查 API 是否有隐藏 API
        if 'hidden_apis' in api and api['hidden_apis']:
            # 对于每个隐藏 API，创建一行
            for hidden_id in api['hidden_apis']:
                if hidden_id in api_id_map:
                    hidden_api = api_id_map[hidden_id]
                    output_description = generate_api_description(hidden_api)

                    # 将输入输出对写入 CSV 文件
                    writer.writerow({'input': input_description, 'output': output_description})
                else:
                    print(f"Hidden API with ID {hidden_id} not found.")
        else:
            # 如果 API 没有隐藏 API，跳过
            pass  # 跳过没有隐藏 API 的 API
