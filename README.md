# RAG-Augmented Chatbot — Hugging Face Space Source

## Project Introduction
This repository contains the **source code for deploying the RAG-augmented chatbot on Hugging Face Spaces**.  
The chatbot retrieves content from FAISS indexes and answers user questions using the OpenAI Chat API.  
It is intended as the deployment companion for the main project.

> The main project repository (with Docker setup and full frontend) is here:  
[Main RAG-Augmented Chatbot Repository](https://github.com/Arsney091289421/RAG-Augmented-chatbot)

---

## Live Demo

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Demo-blue?logo=gradio)](https://huggingface.co/spaces/Daniel192341/RAG-Augmented-chatbot-hfspace)

_(This demo is available on Hugging Face Spaces. Limited usage: 10 requests per session / 100 total per day.)_


---

### Rate Limiting (to protect OpenAI API)

This demo includes a built-in rate limiting mechanism to prevent abuse and protect OpenAI API usage:

- **Per session:** max **10** requests per day  
- **Global total:** max **100** requests per day across all users

All limits reset **automatically every 24 hours**, and are stored using **Redis Cloud** with key expiry (`EXPIRE`) to ensure cleanup.

**Session-level tracking** is implemented via Gradio’s `State` component, allowing browser-based user isolation.

---

## Running Locally 
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

