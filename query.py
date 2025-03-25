import os
import json
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# Load config.json
with open(os.getenv("CONFIG_PATH", "config.json"), 'r') as f:
    config = json.load(f)

TEMPERATURE = config.get("temperature", 0.3)
TOP_K = config.get("top_k", 3)
MODEL_NAME = config.get("model_name", "gpt-3.5-turbo")

print(f"Using model: {MODEL_NAME}, temperature: {TEMPERATURE}, top_k: {TOP_K}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_embeddings_flexible(local_path, hub_filename):
    if os.path.exists(local_path):
        print(f"Loading local: {local_path}")
        if local_path.endswith('.npy'):
            return np.load(local_path)
        elif local_path.endswith('.json'):
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    else:
        print(f"Local not found. Downloading from Hub: {hub_filename}")
        return load_embeddings_from_hub(hub_filename)

def load_faiss_index_flexible(local_path, hub_filename):
    if os.path.exists(local_path):
        print(f"Loading local FAISS index: {local_path}")
        return faiss.read_index(local_path)
    else:
        print(f"Local index not found. Downloading from Hub: {hub_filename}")
        return load_faiss_index_from_hub(hub_filename)

def load_embeddings_from_hub(filename):
    local_path = hf_hub_download(repo_id="Daniel192341/RAG-embeddings-store", filename=filename, repo_type="dataset")
    if filename.endswith('.npy'):
        return np.load(local_path)
    elif filename.endswith('.json'):
        with open(local_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return local_path

def load_faiss_index_from_hub(filename):
    local_path = hf_hub_download(repo_id="Daniel192341/RAG-embeddings-store", filename=filename, repo_type="dataset")
    return faiss.read_index(local_path)

def find_similar_documents(query, model, faiss_index, embeddings, texts, source_label, top_k=5):
    query_embedding = model.encode([query])
    D, I = faiss_index.search(query_embedding, top_k)
    results = [(texts[i], D[0][idx], source_label) for idx, i in enumerate(I[0])]
    return results

def generate_answer_with_gpt(query, context):
    system_prompt = (
        "You are a helpful assistant. "
        "Use ONLY the following provided context to answer the user's question. "
        "If the answer is not found in the context, reply with: "
        "'I don't know based on the provided documents.' "
        "Do not make up information.\n\n"
        f"Context:\n{context}"
    )
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    
    sklearn_embeddings = load_embeddings_flexible("embeddings/sklearn_embeddings.npy", "sklearn_embeddings.npy")
    sklearn_texts = load_embeddings_flexible("embeddings/sklearn_texts.json", "sklearn_texts.json")
    hf_embeddings = load_embeddings_flexible("embeddings/hf_embeddings.npy", "hf_embeddings.npy")
    hf_texts = load_embeddings_flexible("embeddings/hf_texts.json", "hf_texts.json")

    sklearn_index = load_faiss_index_flexible("embeddings/sklearn_index.faiss", "sklearn_index.faiss")
    hf_index = load_faiss_index_flexible("embeddings/hf_index.faiss", "hf_index.faiss")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    user_query = input("Enter your question: ")

    results_sklearn = find_similar_documents(user_query, model, sklearn_index, sklearn_embeddings, sklearn_texts, source_label="sklearn", top_k=TOP_K)
    results_hf = find_similar_documents(user_query, model, hf_index, hf_embeddings, hf_texts, source_label="transformers", top_k=TOP_K)

    combined_context = "\n\n".join([r[0] for r in results_sklearn + results_hf])

    print("\nRetrieved sources:")
    for idx, (text, score, source_label) in enumerate(results_sklearn + results_hf):
        snippet_preview = text[:100].replace('\n', ' ')
        print(f"[{idx+1}] (score={score:.4f}, source={source_label}): {snippet_preview}")

    answer = generate_answer_with_gpt(user_query, combined_context)
    print("\nAnswer:")
    print(answer)
