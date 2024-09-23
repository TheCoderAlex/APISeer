from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import save_directory


model = T5ForConditionalGeneration.from_pretrained('Trained/APISeer_Trained-9-13-base')

tokenizer = T5Tokenizer.from_pretrained(save_directory)

text = "API uses GET method at /highRisk/login."

input_ids = tokenizer('generate api: ' + text, return_tensors='pt').input_ids

outputs = model.generate(
    input_ids,
    max_new_tokens=4000,
    do_sample=True,
    top_k=60,
    top_p=0.89,
    temperature=0.7,
    num_return_sequences=10  # 生成5个不同的回答
)

decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

print("问题："+text)

for i, decoded_output in enumerate(decoded_outputs):
    print(f"回答 {i + 1}: {decoded_output}\n")