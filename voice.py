import os
import json
from IPython.display import Audio, display, update_display
from openai import OpenAI
from huggingface_hub import login
import torch
from transformers import pipeline
import gradio as gr
from dotenv import load_dotenv
from tqdm import tqdm
from litellm import completion
import numpy as np
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from pathlib import Path
import imageio_ffmpeg

os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

os.makedirs("data", exist_ok=True)

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a SpaceX Crew Dragon mission,
based on the mission transcript.
Use the following context to answer the question. If the context doesn't contain the answer, say you don't know.

Context:
{context}
"""

file_path = Path("data/mission_transcript.txt")

load_dotenv(override=True)

MODEL = "gpt-4o-mini" 
DB_NAME = "data/chroma_db"

print("🔄 Program started, libraries are loading...")

login(token=os.getenv("HUGGINGFACE_TOKEN"))

openai = OpenAI()
audio_filename = "data/mission_audio_small.mp4"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    dtype=torch.float16,
    device="cuda",
    return_timestamps=True
)

# 🛠️ Whisper Step (Skipped if transcript exists)
if not file_path.exists():
    print("🎙️ Audio file is being transcribed...")
    try:
        result = pipe(audio_filename)
        tsc = result["text"]
        MAX_CHARS = 15000
        tsc_for_llama = tsc[:MAX_CHARS]

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(tsc_for_llama)
        print("✅ Transcript successfully saved.")
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
else:
    print("📁 Transcript file already exists, Whisper step skipped.")

collection_name = "mission_logs"
embedding_model = "text-embedding-3-large"
avg_chunk_size = 750

class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")
    
    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata
        )

class Chunks(BaseModel):
    chunks: list[Chunk]

def fetch_docs():
    docs = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        docs.append({
            "type": "transcript",
            "source": file_path.as_posix(),
            "text": content
        })
    else:
        return 0
    return docs

docs = fetch_docs()

def make_prompt(document):
    how_many = (len(document["text"]) // avg_chunk_size) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is an official transcript from a SpaceX Crew Dragon space mission.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the mission.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""

def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]

def process_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

def create_chunks(documents):
    chunks = []
    for doc in tqdm(documents):
        chunks.extend(process_document(doc))
    return chunks

print("📦 Text is being split into chunks...")
chunks = create_chunks(docs)

def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    print(f"✅ Vectorstore created, {collection.count()} documents added.")
    return collection 

collection = create_embeddings(chunks)

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]

k = 10

def fetch_context_unranked(question):
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings = [query], n_results=k)
    fetched_chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        fetched_chunks.append(Result(page_content=result[0], metadata=result[1]))
    return fetched_chunks

def fetch_context(question):
    fetched_chunks = fetch_context_unranked(question)
    return rerank(question, fetched_chunks) 

def make_rag_messages(question, history, chunks):
    context = "\n\n".join(f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

def rewrite_query(question, history=[]):
    message = f"""
You are in a conversation with a user, answering questions about a SpaceX Crew Dragon space mission.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the mission.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    query = rewrite_query(question, history)
    print(f"🔍 Search Query: {query}")
    chunks = fetch_context(query)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks

def gradio_chat(message, history):
    formatted_history = []
    for item in history:
        if isinstance(item, dict):
            formatted_history.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                formatted_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                formatted_history.append({"role": "assistant", "content": bot_msg})
            
    answer, _ = answer_question(message, formatted_history)
    return answer

# ⚖️ Advanced Tri-Reliability Judge Function
def run_custom_evaluation():
    print("\n🚀 Custom Eval System Initializing...")
    
    test_path = Path("data/test.jsonl")
    if not test_path.exists():
        print("❌ data/test.jsonl file not found!")
        return
        
    test_cases = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_cases.append(json.loads(line))
            
    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected_answer = case["answer"]
        
        print(f"\n--- Test Question {i}/{len(test_cases)} ---")
        print(f"❓ Question: {question}")
        
        # 🛠️ FIXED: Now we also capture chunks!
        generated_answer, chunks = answer_question(question, [])
        
        # Convert chunk texts to string so the judge can read them
        context_text = "\n---\n".join([c.page_content for c in chunks])
        
        print(f"🤖 Generated Answer: {generated_answer}")
        print(f"🎯 Expected Answer: {expected_answer}")
        
        # Advanced Tri-Reliability Judge Prompt
        judge_prompt = f"""
You are an expert RAG (Retrieval-Augmented Generation) system evaluation judge.
Review the following data and evaluate the system in 3 different categories from 0 to 100:

Question: {question}
Context Retrieved from Database: 
{context_text}

Expected Answer (Ground Truth): {expected_answer}
Generated Answer: {generated_answer}

Please evaluate the following 3 metrics separately and write them in exactly this format:
1. Faithfulness (Were the facts taken only from the context, any hallucinations?): [0-100]
2. Answer_Relevance (Does the answer actually address the question?): [0-100]
3. Context_Precision (Is the retrieved context sufficient to answer the question?): [0-100]
Rationale: [A short 1-2 sentence explanation summarizing all these scores]
"""
        judge_response = completion(model=MODEL, messages=[{"role": "user", "content": judge_prompt}])
        evaluation_result = judge_response.choices[0].message.content
        
        print(f"⚖️ Judge Decision:\n{evaluation_result}")
        print("=" * 60)

    print("✅ All Tests Completed and Reported!\n")

if __name__ == "__main__":
    # If you want, you can run the test first and see the scores in the console:
    run_custom_evaluation()
    
    # Then you can launch the interface:
    print("🚀 Gradio Chatbot Interface Initializing...")
    gr.ChatInterface(fn=gradio_chat).launch(inbrowser=True)