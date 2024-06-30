from chatbot import response
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    answer = response(query)

    return jsonify({'answer': answer})