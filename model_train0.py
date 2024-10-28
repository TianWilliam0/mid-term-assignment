from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch


tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_sequences(sequences):

    return tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)


with open('train_sequences.txt', 'r', encoding='utf-8') as file:
    sequences = file.read().splitlines()


tokenized_data = tokenize_sequences(sequences)


model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.config.pad_token_id = tokenizer.pad_token_id


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print("device is {}".format(device))

dataset = Dataset.from_dict({
    'input_ids': tokenized_data['input_ids'],
    'attention_mask': tokenized_data['attention_mask'],
    'labels': tokenized_data['input_ids']
})

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results_distilgpt2',
    per_device_train_batch_size=16,
    num_train_epochs=50,
    learning_rate=5e-5,
    warmup_steps=100,
    save_steps=1_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    eval_strategy="steps",
    eval_steps=1_000,
    report_to="tensorboard"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset
)


trainer.train()


model.save_pretrained('./trained_model_distilgpt2')
tokenizer.save_pretrained('./trained_model_distilgpt2')

# 生成文本
input_prompt = "郭靖和黄蓉"
inputs = tokenizer(input_prompt, return_tensors='pt').to(device)

# 生成文本
outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
