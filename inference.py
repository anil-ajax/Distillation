import tensorflow as tf
from transformers import GPT2Tokenizer

# Define the custom student model
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

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Initialize the model and load the weights
student_model = CustomStudentModel()
student_model.load_weights("/content/sample_data/m1/custom_student_model")

# Function to generate a response
def generate_response(question):
    inputs = tokenizer.encode(question, return_tensors="tf")
    outputs = student_model(inputs)
    response = tokenizer.decode(tf.argmax(outputs, axis=-1)[0], skip_special_tokens=True)
    return response

# Chatbot loop
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = generate_response(question)
    print(f"Chatbot: {response}")
