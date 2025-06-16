# NeoDoc
AI-Powered Doctor Chatbot 
# 🧠 NeoDoc — AI-Powered Doctor Chatbot

**NeoDoc** is a Generative AI-based medical assistant designed to help users get initial diagnostic insights and first-aid guidance based on their symptoms. It uses **LLMs + RAG (Retrieval-Augmented Generation)** to provide context-aware, reliable, and empathetic responses.

---

## 🚀 Features

- 🔍 Collects symptoms via conversational interface
- 🧾 Retrieves relevant data from trusted medical documents
- 🧠 Generates possible diagnoses using Gemini Pro API
- 🩺 Suggests treatments, self-care tips, and emergency actions
- 🤖 Built with LangChain, Gradio, ChromaDB, and Gemini

---

## 🛠️ Tech Stack

| Component       | Tool/Library             |
|----------------|--------------------------|
| LLM            | Gemini Pro (via API)     |
| Framework      | LangChain                |
| Retrieval      | ChromaDB (Vector DB)     |
| Frontend       | Gradio                   |
| Data Source    | Medical PDFs (preloaded) |
| Deployment     | Local / Web (optional)   |

---

## 📂 Project Structure

NeoDoc/
├── app.py # Gradio chatbot app
├── rag_chain.py # LangChain + RAG pipeline
├── vectorstore/ # ChromaDB storage
├── medical_pdfs/ # Trusted documents (PDFs)
├── utils/ # Utility functions
├── assets/ # Logo/images
└── README.md

pip install -r requirements.txt
