Text Generation Project with distilgpt2 and gpt2 Models
This project uses the distilgpt2 and gpt2 models, fine-tuned on two distinct datasets: a Chinese dataset based on The Condor Trilogy and an English dataset based on Harry Potter. The goal is to generate contextually relevant and stylistically consistent text by training and testing on these datasets.

Before running this project, please create a 3.10 virtual environment and install the requirement lib.

├── pre_pro.py                  # Cleans and segments raw text data
├── create_sequences.py         # Splits segmented text into sequences of fixed length (default 100 chars)
├── model_train.py              # Script for model training
├── model_train0.py             # Additional training script
├── model_train_op.py           # Training script with optional configurations
├── model_exam.py               # Testing script for `gpt2`-based models
├── model_exam_distilgpt2.py    # Testing script for `distilgpt2`-based models
├── model_exam_harry.py         # Testing script for Harry Potter data models
├── result_harry/               # Results using `distilgpt2` trained on *Harry Potter*
├── result_distilgpt2/          # Results using `distilgpt2` trained on *The Condor Trilogy*
├── results/                    # Results using `gpt2` trained on *The Condor Trilogy*
├── trained_model_harry/        # Intermediate model checkpoints on *Harry Potter*
├── trained_model/              # Intermediate model checkpoints on *The Condor Trilogy*
├── trained_model_distilgpt2/   # Intermediate model checkpoints on *The Condor Trilogy*
└── README.md                   # Project documentation
Of these, result_harry、result_distilgpt2、results、trained_model_harry、trained_model、trained_model_distilgpt2 are larger and is stored in the Quark Cloud Driver.
LINK: https://pan.quark.cn/s/0ffdc42df725

1. Datasets
Chinese Dataset: Text from The Condor Trilogy, a classical martial arts series.
English Dataset: Text from the Harry Potter series.
2. Prerequisites
Python 3.10+
transformers library
torch library
GPU (optional, for faster training and inference)
To install required libraries, run:


pip install torch transformers
3. Data Preprocessing
Cleaning and Segmentation
pre_pro.py: Cleans and segments raw text data for both datasets to prepare them for training.
create_sequences.py: Splits the preprocessed text into sequences of specified length (default: 100 characters) for more structured model input.
4. Model Training
Training Scripts
Model Options: Training can be performed using either distilgpt2 or gpt2 models. The three main scripts used are model_train.py, model_train0.py, and model_train_op.py.
Checkpoints: Checkpoints are saved periodically during training in the checkpoint/ directory, allowing model performance tracking and comparison between iterations.
Example training code snippet for using Hugging Face’s distilgpt2:


from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load model and tokenizer
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./checkpoint",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=5000,
    save_total_limit=2
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()
5. Model Testing
Three testing scripts allow evaluation of models on different datasets and configurations:

model_exam.py: Tests the gpt2 model on The Condor Trilogy.
model_exam_distilgpt2.py: Tests distilgpt2 on The Condor Trilogy.
model_exam_harry.py: Tests distilgpt2 on the Harry Potter dataset.
Testing scripts load trained models and output results stored in the result_harry, result_distilgpt2, and results directories.

Text Generation
Example code for generating text using the fine-tuned models:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained('./trained_model_harry')
tokenizer = AutoTokenizer.from_pretrained('./trained_model_harry')

def generate_text(prompt, max_length=2000, temperature=0.6, top_k=40, top_p=0.85, repetition_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors='pt')

    # Use GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        min_length=800,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        early_stopping=False,
        eos_token_id=None,
        bad_words_ids=[[tokenizer.eos_token_id]]
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Test generation
prompt = "Zhang Sanfeng looked at Zhang Wuji and slowly said, “Wuji, the essence of Taiji Sword lies not in the technique itself, but in"
generated_text = generate_text(prompt)
print("Generated Text:")
print(generated_text)
Generation Parameters
max_length: Controls maximum length of generated text.
temperature: Regulates randomness; values between 0.5 and 1.0 provide coherent text.
top_k: Samples from top k words based on probability.
top_p: Nucleus sampling for diverse vocabulary selection.
repetition_penalty: Adds penalty to repetitive content.
6. Results and Evaluation
Generated text results are saved as follows:

result_harry/: Results from distilgpt2 fine-tuned on Harry Potter.
result_distilgpt2/: Results from distilgpt2 fine-tuned on The Condor Trilogy.
results/: Results from gpt2 fine-tuned on The Condor Trilogy.
Exam results:
![image](https://github.com/user-attachments/assets/c187eccb-4ec4-4982-bfc5-e19e1d008416)

7. Troubleshooting Tips
If generated text is too short, consider increasing min_length in the generate function.
If repetitive content appears, adjust repetition_penalty or experiment with top_k and top_p values.
Model checkpoints can be loaded to inspect and compare performance at different training stages.
8. Resources
Transformers Library
Hugging Face Model Hub
