from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model = AutoModelForCausalLM.from_pretrained('./trained_model_harry')
tokenizer = AutoTokenizer.from_pretrained('./trained_model_harry')

# Set pad_token_id to avoid errors during generation
model.config.pad_token_id = model.config.eos_token_id

# Generated text function
def generate_text(prompt, max_length=1000, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1):
    inputs = tokenizer(prompt, return_tensors='pt')

    # GPU
    if torch.cuda.is_available():
        model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,  # Controls the maximum length of generated text
        min_length=10,  # Increase the minimum length to ensure that the generated text is not too short
        temperature=temperature,  # Control the randomness of the generation
        top_k=top_k,  # Limit the generation of the first k words with the highest probability
        top_p=top_p,  # Use nucleus sampling
        repetition_penalty=repetition_penalty,  # Repetition of punishment
        do_sample=True,  # Random sampling
        num_return_sequences=1,  # The number of text sequences returned
        no_repeat_ngram_size=3,  # Prevent duplicate triples
        early_stopping=True,  # Disable early stop
        eos_token_id=None,  # Remove end flag
        bad_words_ids=[[tokenizer.eos_token_id]]  # Disable generating end flag
    )
    # Decode the generated token ids and convert them to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# 测试生成
while True:
    prompt = input("请输入提示语: ")
    generated_text = generate_text(prompt, max_length=100, temperature=0.5, top_k=50, top_p=0.99, repetition_penalty=1.2)
    print("生成的文本:")
    print(generated_text)
