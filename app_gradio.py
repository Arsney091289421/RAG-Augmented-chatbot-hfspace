import os
import json
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import gradio as gr
from query import generate_answer_with_gpt, find_similar_documents

# set env
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# load files from loacl or Hub
def load_embeddings_flexible(local_path, hub_filename):
    if os.path.exists(local_path):
        print(f" Loading local: {local_path}")
        if local_path.endswith('.npy'):
            return np.load(local_path)
        elif local_path.endswith('.json'):
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    else:
        print(f" Downloading from HF hub: {hub_filename}")
        local_path = hf_hub_download(repo_id="Daniel192341/RAG-embeddings-store", filename=hub_filename, repo_type="dataset")
        if hub_filename.endswith('.npy'):
            return np.load(local_path)
        elif hub_filename.endswith('.json'):
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)

def load_faiss_index_flexible(local_path, hub_filename):
    if os.path.exists(local_path):
        print(f" Loading local index: {local_path}")
        return faiss.read_index(local_path)
    else:
        print(f" Downloading index from HF hub: {hub_filename}")
        local_path = hf_hub_download(repo_id="Daniel192341/RAG-embeddings-store", filename=hub_filename, repo_type="dataset")
        return faiss.read_index(local_path)

# load
sklearn_embeddings = load_embeddings_flexible("embeddings/sklearn_embeddings.npy", "sklearn_embeddings.npy")
sklearn_texts = load_embeddings_flexible("embeddings/sklearn_texts.json", "sklearn_texts.json")
hf_embeddings = load_embeddings_flexible("embeddings/hf_embeddings.npy", "hf_embeddings.npy")
hf_texts = load_embeddings_flexible("embeddings/hf_texts.json", "hf_texts.json")
sklearn_index = load_faiss_index_flexible("embeddings/sklearn_index.faiss", "sklearn_index.faiss")
hf_index = load_faiss_index_flexible("embeddings/hf_index.faiss", "hf_index.faiss")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Gradio 
def rag_qa(user_query):
    results_sklearn = find_similar_documents(user_query, model, sklearn_index, sklearn_embeddings, sklearn_texts, source_label="sklearn", top_k=3)
    results_hf = find_similar_documents(user_query, model, hf_index, hf_embeddings, hf_texts, source_label="transformers", top_k=3)

    combined_context = "\n\n".join([r[0] for r in results_sklearn + results_hf])
    answer = generate_answer_with_gpt(user_query, combined_context)

    sources_str = ""
    for idx, (text, score, source) in enumerate(results_sklearn + results_hf):
        snippet_preview = text[:150].replace('\n', ' ')
        sources_str += f"[{idx+1}] Source: {source} | Relevance: {score:.4f}\n{snippet_preview}\n\n"

    return answer, sources_str

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("""
    # RAG Chatbot (scikit-learn + HuggingFace)
    This chatbot answers questions **ONLY** based on retrieved documentation from:
    - [Scikit-learn documentation](https://scikit-learn.org/stable/)
    - [Hugging Face Transformers GitHub](https://github.com/huggingface/transformers)
    """)
    with gr.Row():
        query_input = gr.Textbox(label="Ask your question")
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")
    with gr.Row():
        answer_output = gr.Textbox(label="Answer", lines=4)
        sources_output = gr.Textbox(label="Top Retrieved Sources", lines=10)
    submit_btn.click(rag_qa, inputs=query_input, outputs=[answer_output, sources_output])
    clear_btn.click(lambda: ("", ""), outputs=[answer_output, sources_output])

demo.launch()
