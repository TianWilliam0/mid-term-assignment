from transformers import GPT2LMHeadModel, GPT2Tokenizer
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    # Convert prompt to model input format
    inputs = tokenizer(prompt, return_tensors='pt')

    # Call the model to generate text
    outputs = model.generate(
        inputs['input_ids'],  # Enter the token ids
        attention_mask=inputs['attention_mask'],  # Use attention mask
        max_length=max_length,  # The maximum length of the generated text
        temperature=temperature,  # Control the randomness of the generation
        top_k=top_k,  # Limit the generation of the first k words with the highest probability
        top_p=top_p,  # Use nucleus sampling
        do_sample=True,  # Whether to use random sampling (rather than greedy searching)
        num_return_sequences=1  # The number of text sequences returned
    )

    # Decode the generated token ids and convert them to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


# Load the saved model and word divider
model = GPT2LMHeadModel.from_pretrained('./trained_model')
tokenizer = GPT2Tokenizer.from_pretrained('./trained_model')

# Belay pad_token_The id is set to eos_token_id to avoid errors during generation
model.config.pad_token_id = model.config.eos_token_id
# Test generated text
prompt = "张三丰注视着张无忌，缓缓说道：“无忌，太极剑的精髓不在于剑法本身，而在于"
generated_text = generate_text(prompt, max_length=400, temperature=1, top_k=50, top_p=0.9)

print("Generated Text:")
print(generated_text)
