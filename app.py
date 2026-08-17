from flask import Flask, request, jsonify
from flask_cors import CORS
from services.summarization import IntelligentLegalPipeline
from routes.summarization_routes import summarization

app = Flask(__name__)
CORS(app)  

pipeline = IntelligentLegalPipeline()

app.register_blueprint(summarization, url_prefix='/api/legal')

@app.route('/')
def home():
    return {"message": "Server is running successfully!"}


if __name__ == '__main__':
    app.run(debug=True, port=5000)