from flask import Flask, request, jsonify
from flask_cors import CORS
from services.summarization import IntelligentLegalPipeline
from routes.summarization_routes import legal_summarization
from routes.law_and_jurisprudence_search_routes import legal_search_bp

app = Flask(__name__)
CORS(app)  

pipeline = IntelligentLegalPipeline()

app.register_blueprint(legal_summarization, url_prefix='/api/legal')
app.register_blueprint(legal_search_bp, url_prefix='/api/legal')

@app.route('/')
def home():
    return {"message": "Server is running successfully!"}


if __name__ == '__main__':
    app.run(debug=True, port=5000)