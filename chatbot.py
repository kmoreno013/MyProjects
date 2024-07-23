from flask import Flask, jsonify, request, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Initialize model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        input_text = data.get('prompt', '')

        # Create conversation history string
        history = "\n".join(conversation_history)

        # Tokenize the input text and history
        inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

        # Generate the response from the model
        outputs = model.generate(**inputs, max_length=60)

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Add interaction to conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
