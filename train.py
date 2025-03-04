!pip install datasets # we will use wikitext dataset

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# gpt2 as teacher model
teacher_model_name = "gpt2"
teacher_model = TFGPT2LMHeadModel.from_pretrained(teacher_model_name)
teacher_tokenizer = GPT2Tokenizer.from_pretrained(teacher_model_name)

# Define student model
class CustomStudentModel(tf.keras.Model):
    def __init__(self):
        super(CustomStudentModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=50257, output_dim=128)
        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True)
        self.dense = tf.keras.layers.Dense(50257)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        return self.dense(x)

student_model = CustomStudentModel()

# distillation loss function
def distillation_loss(y_true, y_pred, teacher_logits, temperature=2.0):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_probs = tf.nn.softmax(y_pred / temperature, axis=-1)
    return tf.reduce_mean(tf.keras.losses.KLDivergence()(teacher_probs, student_probs))

# Prepare
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize_function(examples):
    return teacher_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128, return_tensors="tf")

# teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print(f"Tokenizer vocabulary size: {teacher_tokenizer.vocab_size}")

print(f"Tokenizer vocabulary size with special tokens: {len(teacher_tokenizer)}")

teacher_tokenizer.pad_token = teacher_tokenizer.eos_token # I was facing pad errors

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.shuffle().batch(8)

# Compile student model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

@tf.function
def train_step(student_inputs, teacher_logits):
    with tf.GradientTape() as tape:
        student_logits = student_model(student_inputs)
        loss = distillation_loss(student_inputs, student_logits, teacher_logits)
    gradients = tape.gradient(loss, student_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
    return loss

# Tokenize the dataset
def tokenize_function(examples):
    return teacher_tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )

# Training loop
epochs = 1
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in tokenized_dataset:
        inputs = np.array(batch["input_ids"])
        with tf.GradientTape() as tape:
            teacher_logits = teacher_model(inputs).logits
        loss = train_step(inputs, teacher_logits)
        print(f"Loss: {loss.numpy()}")

# Save the student model
student_model.save_weights("./custom_student_model")
    question = input("You: ")
    if question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = generate_response(question)
    print(f"Chatbot: {response}")
