## RAG-Augmented Chatbot

### Project Introduction
A lightweight RAG chatbot based on FAISS local retrieval and OpenAI Chat API. It can answer user questions by retrieving content from HuggingFace and scikit-learn official documentation, combining the context, and generating accurate answers using GPT models.

> Default model is GPT-3.5-Turbo. You can modify `query.py` to change to any OpenAI-supported model (e.g., GPT-4).

---

### Demo Screenshots

**Homepage**  

**Example Q&A**  

---

### Demo Video

You can preview the demonstration video here:

[![Watch the demo video](https://img.youtube.com/vi/TTDYCGNy000/0.jpg)](https://youtu.be/TTDYCGNy000)

---

### Tech Stack
- Python
- FAISS
- Sentence-Transformers
- OpenAI Chat API
- Flask
- Docker deployment

---

### Features
- Precomputed document embeddings (stored locally in npy + json + faiss index files)
- Multi-source document retrieval (HuggingFace & Scikit-learn)
- Similarity scoring and context combination
- Answer generation via GPT model
- Simple frontend to display answers, sources, and relevance scores

---

### Local Setup
#### 1. Clone the repository
```bash
git clone https://github.com/Arsney091289421/RAG-Augmented-chatbot.git
cd RAG-Augmented-chatbot
```
#### 2. Install dependencies
```bash
pip install -r requirements.txt
```
#### 3. Set environment variables
Create a `.env` file in the project root with the following content:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
#### 4. Run the app
```bash
python app.py
```
Visit [http://localhost:5050](http://localhost:5050) in your browser.

---

### Optional Docker Deployment
```bash
docker build -t rag-chatbot .
docker run -p 5050:5050 rag-chatbot
```

---

### Contact
- GitHub: [https://github.com/Arsney091289421](https://github.com/Arsney091289421)
