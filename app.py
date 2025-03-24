from flask import Flask, request, jsonify, send_from_directory
from query import generate_answer_with_gpt, find_similar_documents, load_embeddings, load_faiss_index
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load models and indexes once at startup
sklearn_embeddings, sklearn_texts = load_embeddings("embeddings/sklearn_embeddings.npy", "embeddings/sklearn_texts.json")
hf_embeddings, hf_texts = load_embeddings("embeddings/hf_embeddings.npy", "embeddings/hf_texts.json")
sklearn_index = load_faiss_index("embeddings/sklearn_index.faiss")
hf_index = load_faiss_index("embeddings/hf_index.faiss")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("question", "")

    # Find similar docs with source labels
    results_sklearn = find_similar_documents(user_query, model, sklearn_index, sklearn_embeddings, sklearn_texts, source_label="sklearn", top_k=3)
    results_hf = find_similar_documents(user_query, model, hf_index, hf_embeddings, hf_texts, source_label="transformers", top_k=3)

    combined_context = "\n\n".join([r[0] for r in results_sklearn + results_hf])

    # Generate GPT answer
    answer = generate_answer_with_gpt(user_query, combined_context)

    # Return both answer and sources
    sources = [
    {
        "snippet": r[0][:120].replace('\n', ' '),  # replace linebreaks
        "source": r[2],                           # source code - sklearn / transformers)
        "relevance_score": round(float(r[1]), 4)         # 4 decimal points and float - flask - jsonify() only support float
    } 
    for r in (results_sklearn + results_hf)
]
    return jsonify({
        "answer": answer,
        "sources": sources
    })

@app.route("/")
def home():
    return send_from_directory("templates", "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
