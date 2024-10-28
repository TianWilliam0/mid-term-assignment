from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model = GPT2LMHeadModel.from_pretrained('./trained_model_harry')
tokenizer = GPT2Tokenizer.from_pretrained('./trained_model_harry')


model.config.pad_token_id = model.config.eos_token_id


def generate_text(prompt, max_length=1000, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1):
    inputs = tokenizer(prompt, return_tensors='pt')


    if torch.cuda.is_available():
        model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        min_length=10,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        early_stopping=True,
        eos_token_id=None,
        bad_words_ids=[[tokenizer.eos_token_id]]
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


while True:
    prompt = input("请输入提示语: ")
    generated_text = generate_text(prompt, max_length=100, temperature=0.5, top_k=50, top_p=0.99, repetition_penalty=1.2)
    print("生成的文本:")
    print(generated_text)
