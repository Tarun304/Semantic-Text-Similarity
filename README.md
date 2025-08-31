# Text Similarity API

A robust, production-ready REST API for computing semantic similarity between text documents using state-of-the-art natural language processing techniques.

## 🚀 Overview

This project provides a FastAPI-based web service that calculates semantic similarity scores between two text inputs. It leverages the BAAI/bge-large-en-v1.5 embedding model from Hugging Face to generate high-quality text embeddings and compute cosine similarity scores.

### Key Features

- **Semantic Understanding**: Uses advanced NLP techniques to understand meaning, not just word overlap
- **Robust Text Preprocessing**: Comprehensive text cleaning including HTML removal, URL filtering, and lemmatization
- **Production Ready**: Built with FastAPI for high performance and automatic API documentation
- **Cloud Deployable**: Configured for deployment on Render with proper environment management
- **Comprehensive Logging**: Detailed logging with Loguru for monitoring and debugging
- **Error Handling**: Robust error handling and validation throughout the application

## 🏗️ Architecture

The project follows a clean, modular architecture:

```
src/
├── api/                 # FastAPI application layer
│   ├── app.py          # Main FastAPI application
│   ├── routes.py       # API endpoint definitions
│   └── models.py       # Pydantic data models
├── backend/            # Core business logic
│   └── similarity.py   # Semantic similarity engine
└── utils/              # Shared utilities
    └── logger.py       # Logging configuration
```

### Technology Stack

- **Framework**: FastAPI (Python web framework)
- **NLP Libraries**: NLTK, SpaCy
- **Embedding Model**: BAAI/bge-large-en-v1.5 (via Hugging Face API)
- **Logging**: Loguru
- **Deployment**: Render (cloud platform)
- **Environment**: Python 3.x with virtual environment

## 📋 Prerequisites

Before running this project, ensure you have:

- Python 3.8 or higher
- A Hugging Face API token
- Internet connection (for model downloads and API calls)

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Tarun304/Semantic-Text-Similarity.git
cd Semantic-Text-Similarity
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

**Getting a Hugging Face Token:**
1. Visit [Hugging Face](https://huggingface.co/)
2. Create an account or sign in
3. Go to Settings → Access Tokens
4. Create a new token with read permissions

### 5. Download Required Models

The application will automatically download required NLTK and SpaCy models on first run, or you can manually install them:

```bash
# NLTK models
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# SpaCy model
python -m spacy download en_core_web_sm
```

## 🚀 Running the Application

### Development Mode

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base URL**: `http://localhost:8000`
- **Interactive Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/`

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Endpoints

#### 1. Health Check
- **URL**: `/`
- **Method**: `GET`
- **Description**: Check if the API is running
- **Response**:
```json
{
  "message": "Text Similarity API is running 🚀"
}
```

#### 2. Calculate Similarity
- **URL**: `/api/similarity`
- **Method**: `POST`
- **Description**: Calculate semantic similarity between two texts

**Request Body:**
```json
{
  "text1": "The quick brown fox jumps over the lazy dog",
  "text2": "A fast auburn fox leaps across a sleepy canine"
}
```

**Response:**
```json
{
  "similarity score": 0.8542
}
```

**Response Format:**
- `similarity score`: Float between 0 and 1, where:
  - `0.0` = Completely different meanings
  - `1.0` = Identical or very similar meanings

### Example Usage

#### Using curl
```bash
curl -X POST "http://localhost:8000/api/similarity" \
     -H "Content-Type: application/json" \
     -d '{
       "text1": "The weather is sunny today",
       "text2": "It is a beautiful day with clear skies"
     }'
```

#### Using Python requests
```python
import requests

url = "http://localhost:8000/api/similarity"
data = {
    "text1": "The weather is sunny today",
    "text2": "It is a beautiful day with clear skies"
}

response = requests.post(url, json=data)
result = response.json()
print(f"Similarity Score: {result['similarity score']}")
```

## 🔧 How It Works

### Text Preprocessing Pipeline

1. **Lowercase Conversion**: Converts all text to lowercase
2. **HTML Tag Removal**: Strips HTML tags using regex
3. **URL Removal**: Removes URLs and web addresses
4. **Punctuation Removal**: Strips all punctuation marks
5. **Tokenization**: Splits text into individual words
6. **Stopword Removal**: Removes common English stopwords
7. **Lemmatization**: Reduces words to their base form using SpaCy

### Semantic Similarity Calculation

1. **Text Embedding**: Uses BAAI/bge-large-en-v1.5 model to generate 1024-dimensional embeddings
2. **Normalization**: Normalizes embeddings to unit vectors
3. **Cosine Similarity**: Computes cosine similarity between normalized embeddings
4. **Score Rounding**: Rounds the final score to 4 decimal places

### Model Details

- **Embedding Model**: BAAI/bge-large-en-v1.5
- **Embedding Dimensions**: 1024
- **Similarity Metric**: Cosine Similarity
- **Preprocessing**: NLTK + SpaCy pipeline

## 🌐 Deployment

### Deploying to Render

This project is configured for easy deployment on Render:

1. **Fork/Clone** the repository to your GitHub account
2. **Connect** your repository to Render
3. **Set Environment Variables**:
   - `HUGGINGFACEHUB_API_TOKEN`: Your Hugging Face API token
4. **Deploy**: Render will automatically build and deploy using the provided `render.yaml`

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face API token for model access | Yes |
| `PORT` | Port number (set automatically by Render) | No |

### Deployment Files

- `Procfile`: Defines the web process for Render
- `render.yaml`: Render deployment configuration
- `requirements.txt`: Python dependencies

## 🧪 Testing

### Manual Testing

1. Start the application
2. Visit `http://localhost:8000/docs`
3. Use the interactive Swagger UI to test endpoints
4. Try different text combinations to verify similarity scores

### Example Test Cases

```python
# High similarity
text1 = "The cat is on the mat"
text2 = "A feline is sitting on the carpet"

# Medium similarity  
text1 = "I love programming in Python"
text2 = "Coding with Python is enjoyable"

# Low similarity
text1 = "The weather is sunny today"
text2 = "Quantum physics explains particle behavior"
```

## 📊 Performance Considerations

- **API Response Time**: Typically 2-5 seconds per request (depends on text length and API latency)
- **Memory Usage**: ~500MB RAM (includes SpaCy model)
- **Concurrent Requests**: Limited by Hugging Face API rate limits
- **Text Length**: Handles texts up to 512 tokens effectively

## 🔍 Troubleshooting

### Common Issues

1. **SpaCy Model Not Found**
   ```
   Error: SpaCy model 'en_core_web_sm' not found
   Solution: python -m spacy download en_core_web_sm
   ```

2. **Missing Hugging Face Token**
   ```
   Error: Hugging Face API token not found
   Solution: Set HUGGINGFACEHUB_API_TOKEN in .env file
   ```

3. **NLTK Data Missing**
   ```
   Error: NLTK data not found
   Solution: The app will auto-download, or manually run:
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. **API Rate Limiting**
   ```
   Error: Hugging Face API rate limit exceeded
   Solution: Wait and retry, or upgrade Hugging Face plan
   ```

### Logs

The application uses structured logging with Loguru. Check logs for:
- Request processing details
- Error messages and stack traces
- Performance metrics
- API call status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all classes and methods
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the BAAI/bge-large-en-v1.5 model
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [SpaCy](https://spacy.io/) for advanced NLP capabilities
- [NLTK](https://www.nltk.org/) for text processing utilities
- [Render](https://render.com/) for cloud deployment platform

## 📞 Support

For support, questions, or feature requests:

1. Check the [Issues](../../issues) page
2. Review the API documentation at `/docs`
3. Contact the development team

---

**Built with ❤️ using FastAPI and modern NLP techniques**
