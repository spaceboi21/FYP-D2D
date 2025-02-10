from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, List, Dict

import os
from dotenv import load_dotenv
import base64
import pandas as pd
from io import StringIO

from pymongo import MongoClient

# Pinecone / LangChain
from pinecone import Pinecone as PC, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# For splitting CSV text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # or store in .env

# ---------------------------------------------------
# 1) MongoDB for Chat History
# ---------------------------------------------------
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["fyp_db"]
collection = mongo_db["chat_history"]  # "chat_history" collection

# ---------------------------------------------------
# 2) Pinecone Initialization
# ---------------------------------------------------
pc = PC(api_key=PINECONE_API_KEY)
index_name = "fyp"  # 768 dims, metric=cosine
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ---------------------------------------------------
# FastAPI App + CORS
# ---------------------------------------------------
app = FastAPI(title="LangChain Pinecone Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Helper: Chat History in Mongo
# ---------------------------------------------------
def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    doc = collection.find_one({"session_id": session_id})
    if doc:
        return doc.get("chat_history", [])
    return []

def save_chat_history(session_id: str, chat_history: List[Dict[str, str]]):
    collection.update_one(
        {"session_id": session_id},
        {"$set": {"chat_history": chat_history}},
        upsert=True
    )

# ---------------------------------------------------
# Pydantic Models
# ---------------------------------------------------
class UploadCSVModel(BaseModel):
    session_id: str
    csv_base64: str

class AskModel(BaseModel):
    input: str
    session_id: str

# ---------------------------------------------------
# 3) Pinecone VectorStore + Retriever
# ---------------------------------------------------
def load_db():
    store = PineconeVectorStore(
        index_name="fyp",
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    )
    return store

def create_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    
    vector_db = load_db()
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.8}
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )
    tool = create_retriever_tool(
        compression_retriever,
        name="tool_name",
        description="Retriever tool to query CSV data from Pinecone."
    )
    return tool

# ---------------------------------------------------
# 4) Agent with streaming LLM
# ---------------------------------------------------
def generate_agent_executor():
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4",   # or "gpt-4o" if you have a custom name
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to a retriever tool."),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    tool = create_retriever()
    agent = create_tool_calling_agent(llm, tools=[tool], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[tool],
        verbose=True,
        return_intermediate_steps=True
    )
    return agent_executor

# ---------------------------------------------------
# 5) /upload_csv Endpoint -> upsert CSV to Pinecone
# ---------------------------------------------------
@app.post("/upload_csv")
def upload_csv(payload: UploadCSVModel):
    """
    Expects JSON like:
    {
      "session_id": "...",
      "csv_base64": "base64 CSV data"
    }
    Decodes + splits CSV, upserts into 'fyp' Pinecone index.
    """
    try:
        raw_b64 = payload.csv_base64
        decoded = base64.b64decode(raw_b64)
        csv_str = decoded.decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(csv_str))
        if df.empty:
            raise ValueError("CSV is empty or invalid.")

        # Convert entire CSV to text, chunk it, then store
        csv_text = df.to_csv(index=False)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_text(csv_text)
        docs = [Document(page_content=ch) for ch in chunks]

        store = load_db()  # PineconeVectorStore
        store.add_documents(docs)
        return {"status":"success", "rows_ingested": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------------------------------------------------
# 6) /ask Endpoint -> streaming response
# ---------------------------------------------------
@app.post("/ask")
def ask_question(data: AskModel):
    """
    Expects JSON:
    {
      "input": "...",
      "session_id": "..."
    }
    Returns a streaming text/event-stream response with partial tokens.
    """
    session_id = data.session_id
    user_query = data.input

    # Load chat history from Mongo
    chat_history = get_chat_history(session_id)

    agent_executor = generate_agent_executor()

    async def token_generator():
        ai_response = ""
        async for event in agent_executor.astream_events(
            {"input": user_query, "chat_history": chat_history},
            version="v1"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    ai_response += content
                    yield content
        
        # Save chat logs
        chat_history.append({"role":"user", "content": user_query})
        chat_history.append({"role":"assistant", "content": ai_response})
        save_chat_history(session_id, chat_history)

    return StreamingResponse(token_generator(), media_type="text/event-stream")
