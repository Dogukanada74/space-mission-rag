# 🚀 Space Mission RAG & LLM Ops Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) and LLM Operations (LLMOps) system built to process, index, and query official SpaceX Crew Dragon mission transcripts and audio logs.

## 🌟 Features
- **Speech-to-Text Pipeline:** Automatically transcribes mission audio using Hugging Face's `openai/whisper-small` running on GPU.
- **Structured Data Extraction:** Uses Pydantic schemas with `litellm` and OpenAI models to intelligently chunk and parse transcripts.
- **Persistent Vector Search:** Embeds text chunks via `text-embedding-3-large` and stores them using a persistent `ChromaDB` client.
- **Reranking System:** Re-ranks retrieved chunks dynamically using an LLM-based custom reranker to maximize context precision.
- **LLM-as-a-Judge Evaluation:** Includes an automated evaluation harness assessing Faithfulness, Answer Relevance, and Context Precision.
- **Interactive UI:** Powered by Gradio for real-time conversational querying.

## 🛠️ Tech Stack
- **Python**
- **PyTorch & Hugging Face Transformers** (Whisper ASR)
- **OpenAI API & LiteLLM**
- **ChromaDB** (Vector Database)
- **Pydantic** (Data Validation & Structured Outputs)
- **Gradio** (Web UI)

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/Dogukanada74/space-mission-rag.git](https://github.com/Dogukanada74/space-mission-rag.git)
   cd space-mission-rag
