from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize_sequences(sequences):
    # turn token ids
    return tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)

with open('train_sequences.txt', 'r', encoding='utf-8') as file:
    sequences = file.read()
tokenized_data = tokenize_sequences(sequences)
# print(tokenized_data)
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.eos_token_id

dataset = Dataset.from_dict({
    'input_ids': tokenized_data['input_ids'],
    'attention_mask': tokenized_data['attention_mask'],
    'labels': tokenized_data['input_ids']
})

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    num_train_epochs=2000,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
input_prompt = "郭靖和黄蓉"
inputs = tokenizer(input_prompt, return_tensors='pt')

# outputs = model.generate(inputs['input_ids'], max_length=200)
outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],  # Add attention_mask
    max_length=100,
    temperature=0.7,  # Control the randomness of the generation
    top_k=50,         # Limit the generation of the first k words with the highest probability
    top_p=0.9         # Use nucleus sampling
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
