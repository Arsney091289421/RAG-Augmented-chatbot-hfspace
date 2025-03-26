# RAG-Augmented Chatbot — Hugging Face Space Source

## Project Introduction
This repository contains the **source code for deploying the RAG-augmented chatbot on Hugging Face Spaces**.  
The chatbot retrieves content from FAISS indexes and answers user questions using the OpenAI Chat API.  
It is intended as the deployment companion for the main project.

> The main project repository (with Docker setup and full frontend) is here:  
[Main RAG-Augmented Chatbot Repository](https://github.com/Arsney091289421/RAG-Augmented-chatbot)

---

## Live Demo (currently disabled due to OpenAI API misuse concerns)
You can try the chatbot live on Hugging Face Spaces:  

[![Open In Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/Daniel192341/RAG-Augmented-chatbot-hfspace)

---

## Running Locally (optional)
If you want to test this HF Space repo locally:  

```bash
git clone https://github.com/Arsney091289421/RAG-Augmented-chatbot-hfspace.git
cd RAG-Augmented-chatbot-hfspace
pip install -r requirements.txt
```
Set environment variables
Copy the example file and create your own `.env`:
```bash
cp .env.example .env
```
Then edit `.env` and add your OpenAI API Key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

Then run:
```bash
python app_gradio.py
```

---

## License
MIT License

