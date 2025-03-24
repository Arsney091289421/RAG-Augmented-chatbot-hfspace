# RAG-Augmented Chatbot — Hugging Face Space Source

## Project Introduction
This repository contains the **source code for deploying the RAG-augmented chatbot on Hugging Face Spaces**.  
The chatbot retrieves content from FAISS indexes and answers user questions using the OpenAI Chat API.  
It is intended as the deployment companion for the main project.

> The main project repository (with Docker setup and full frontend) is here:  
[Main RAG-Augmented Chatbot Repository](https://github.com/Arsney091289421/RAG-Augmented-chatbot)

---

## Live Demo
You can try the chatbot live on Hugging Face Spaces:  

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-spaces-lg.svg)](https://huggingface.co/spaces/Daniel192341/RAG-Augmented-chatbot-hfspace)

---

## Running Locally (optional)
If you want to test this HF Space repo locally:  

```bash
git clone https://github.com/Arsney091289421/RAG-Augmented-chatbot-hfspace.git
cd RAG-Augmented-chatbot-hfspace
pip install -r requirements.txt
```
> **Create a `.env` file with your `OPENAI_API_KEY` before running.**  

Then run:
```bash
python app_gradio.py
```

---

## License
MIT License

