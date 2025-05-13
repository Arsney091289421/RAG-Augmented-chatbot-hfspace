import os
import json
import faiss
import numpy as np
import openai
from datetime import date
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import gradio as gr
import redis
from uuid import uuid4

from query import generate_answer_with_gpt, find_similar_documents

# Load environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Redis connection
redis_url = os.getenv("REDIS_URL")
r = redis.from_url(redis_url, decode_responses=True)

# Rate limit function
def check_rate_limit(session_id: str):
    today = date.today().isoformat()
    g_key = f"global:{today}"
    s_key = f"session:{session_id}:{today}"

    g_cnt = r.incr(g_key)
    if g_cnt == 1:
        r.expire(g_key, 86400)

    s_cnt = r.incr(s_key)
    if s_cnt == 1:
        r.expire(s_key, 86400)

    if g_cnt > 100:
        return False, "⚠️ Daily global request limit (100) reached. Try again tomorrow."
    if s_cnt > 10:
        return False, "⚠️ You've reached the 10-request daily limit for this session."
    return True, None

# Embedding + FAISS loading functions
def load_embeddings_flexible(local_path, hub_filename):
    if os.path.exists(local_path):
        if local_path.endswith('.npy'):
            return np.load(local_path)
        elif local_path.endswith('.json'):
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    else:
        local_path = hf_hub_download(repo_id="Daniel192341/RAG-embeddings-store", filename=hub_filename, repo_type="dataset")
        if hub_filename.endswith('.npy'):
            return np.load(local_path)
        elif hub_filename.endswith('.json'):
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)

def load_faiss_index_flexible(local_path, hub_filename):
    if os.path.exists(local_path):
        return faiss.read_index(local_path)
    else:
        local_path = hf_hub_download(repo_id="Daniel192341/RAG-embeddings-store", filename=hub_filename, repo_type="dataset")
        return faiss.read_index(local_path)

# Load models and indexes
sklearn_embeddings = load_embeddings_flexible("embeddings/sklearn_embeddings.npy", "sklearn_embeddings.npy")
sklearn_texts = load_embeddings_flexible("embeddings/sklearn_texts.json", "sklearn_texts.json")
hf_embeddings = load_embeddings_flexible("embeddings/hf_embeddings.npy", "hf_embeddings.npy")
hf_texts = load_embeddings_flexible("embeddings/hf_texts.json", "hf_texts.json")
sklearn_index = load_faiss_index_flexible("embeddings/sklearn_index.faiss", "sklearn_index.faiss")
hf_index = load_faiss_index_flexible("embeddings/hf_index.faiss", "hf_index.faiss")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# RAG QA Function
def rag_qa_with_session(user_query, session_id):
    ok, err_msg = check_rate_limit(session_id)
    if not ok:
        return err_msg, ""

    results_sklearn = find_similar_documents(user_query, model, sklearn_index, sklearn_embeddings, sklearn_texts, source_label="sklearn", top_k=3)
    results_hf = find_similar_documents(user_query, model, hf_index, hf_embeddings, hf_texts, source_label="transformers", top_k=3)

    combined_context = "\n\n".join([r[0] for r in results_sklearn + results_hf])
    answer = generate_answer_with_gpt(user_query, combined_context)

    sources_str = ""
    for idx, (text, score, source) in enumerate(results_sklearn + results_hf):
        snippet = text[:150].replace('\n', ' ')
        sources_str += f"[{idx+1}] Source: {source} | Relevance: {score:.4f}\n{snippet}\n\n"

    return answer, sources_str

# Gradio UI with per-user session_id
with gr.Blocks() as demo:
    session_id = gr.State()

    def generate_session_id():
        return str(uuid4())

    demo.load(fn=generate_session_id, outputs=session_id)

    gr.Markdown("""
    # # RAG Chatbot (scikit-learn + HuggingFace)
    This chatbot answers questions **ONLY** based on retrieved documentation from:
    - [Scikit-learn documentation](https://scikit-learn.org/stable/)
    - [Hugging Face Transformers GitHub](https://github.com/huggingface/transformers)

    ⚠️ **Daily limit**: 10 requests per session / 100 total
                
    Each session is browser-based and automatically resets when reopened or refreshed.
    """)

    input_text = gr.Textbox(label="Ask your question")
    output_answer = gr.Textbox(label="Answer", lines=4)
    output_sources = gr.Textbox(label="Top Retrieved Sources", lines=10)
    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    submit.click(fn=rag_qa_with_session, inputs=[input_text, session_id], outputs=[output_answer, output_sources])
    clear.click(fn=lambda: ("", ""), outputs=[output_answer, output_sources])

demo.launch()
