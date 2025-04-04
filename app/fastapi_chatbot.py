from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
import datetime

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # or store in .env

DASHBOARD_FOLDER = "saved_dashboards"  # Folder to save user dashboards

# ---------------------------------------------------
# 1) MongoDB for Chat History
# ---------------------------------------------------
# Get MongoDB connection string from environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "fyp_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_history")

try:
    # Try to connect to MongoDB Atlas first
    mongo_client = MongoClient(MONGODB_URI)
    # Ping the server to check connection
    mongo_client.admin.command('ping')
    print("Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")
    # Fall back to local MongoDB if available
    try:
        mongo_client = MongoClient("mongodb://localhost:27017/")
        print("Connected to local MongoDB successfully!")
    except Exception as local_e:
        print(f"Error connecting to local MongoDB: {local_e}")
        # Create a simple in-memory fallback for demo purposes
        class MemoryStore:
            def __init__(self):
                self.data = {}
            
            def find_one(self, query):
                user_id = query.get("user_id")
                return self.data.get(user_id, {"chat_history": []})
            
            def update_one(self, query, update, upsert=False):
                user_id = query.get("user_id")
                chat_history = update.get("$set", {}).get("chat_history", [])
                self.data[user_id] = {"chat_history": chat_history}
        
        mongo_client = None
        mongo_db = None
        collection = MemoryStore()
        print("Using in-memory store as fallback")

# If we have a real MongoDB connection, set up the database and collection
if mongo_client:
    mongo_db = mongo_client[MONGODB_DB]
    collection = mongo_db[MONGODB_COLLECTION]

# Create the DASHBOARD_FOLDER if it doesn't exist
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

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
def get_chat_history(user_id: str) -> List[Dict[str, str]]:
    """
    Retrieve chat history for a specific user from MongoDB
    """
    try:
        doc = collection.find_one({"user_id": user_id})
        return doc.get("chat_history", []) if doc else []
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def save_chat_history(user_id: str, chat_history: List[Dict[str, str]]):
    """
    Save chat history for a specific user in MongoDB
    """
    try:
        collection.update_one(
            {"user_id": user_id},
            {"$set": {"chat_history": chat_history}},
            upsert=True
        )
    except Exception as e:
        print(f"Error saving chat history: {e}")

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
# @app.post("/ask")
# def ask_question(data: Dict[str, Any]):
#     """
#     Updated to use user_id for chat history and context
#     """
#     user_id = data.get("user_id")
#     user_query = data.get("input")

#     if not user_id:
#         raise HTTPException(status_code=400, detail="User ID is required")

#     # Load chat history from Mongo
#     chat_history = get_chat_history(user_id)


#     agent_executor = generate_agent_executor()

#     async def token_generator():
#         ai_response = ""
#         # async for event in agent_executor.astream_events(
#         #     {"input": user_query, "chat_history": chat_history},
#         #     version="v1"
#         # ):
#         async for event in agent_executor.astream_events(
#             {"input": user_query, "chat_history": chat_history if chat_history else []},
#             version="v1"
#         ):

#             kind = event["event"]
#             if kind == "on_chat_model_stream":
#                 content = event["data"]["chunk"].content
#                 if content:
#                     ai_response += content
#                     yield content
        
#         # Save chat logs
#         chat_history.append({"role":"user", "content": user_query})
#         chat_history.append({"role":"assistant", "content": ai_response})
#         save_chat_history(session_id, chat_history)

#     return StreamingResponse(token_generator(), media_type="text/event-stream")


# Update the AskModel to be more flexible
class AskModel(BaseModel):
    input: str
    user_id: str = Field(..., description="Unique identifier for the user")
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

# @app.post("/ask")
# async def ask_question(data: AskModel):
#     """
#     Expects JSON:
#     {
#       "input": "User's query",
#       "user_id": "unique_user_identifier",
#       "session_id": "optional_session_id",
#       "chat_history": [optional list of previous messages]
#     }
#     Returns a streaming text/event-stream response with partial tokens.
#     """
#     try:
#         # Validate input
#         if not data.input:
#             raise HTTPException(status_code=400, detail="Input query is required")
        
#         if not data.user_id:
#             raise HTTPException(status_code=400, detail="User ID is required")

#         # Use user_id for chat history
#         user_id = data.user_id
#         user_query = data.input

#         # Load or use provided chat history
#         chat_history = data.chat_history or get_chat_history(user_id)

#         # Convert chat history to LangChain format if needed
#         formatted_chat_history = [
#             {"role": entry.get("role", ""), "content": entry.get("content", "")} 
#             for entry in chat_history
#         ]

#         # Generate agent executor (reuse your existing method)
#         agent_executor = generate_agent_executor()

#         async def token_generator():
#             ai_response = ""
#             try:
#                 # Stream events from agent
#                 async for event in agent_executor.astream_events(
#                     {
#                         "input": user_query, 
#                         "chat_history": formatted_chat_history
#                     },
#                     version="v1"
#                 ):
#                     kind = event.get("event")
#                     if kind == "on_chat_model_stream":
#                         # Check if chunk is a dictionary or an AIMessageChunk object
#                         chunk_data = event.get("data", {}).get("chunk", {})
                        
#                         # Handle different chunk formats
#                         if hasattr(chunk_data, "content"):
#                             # If it's an AIMessageChunk object
#                             content = chunk_data.content
#                         elif isinstance(chunk_data, dict):
#                             # If it's a dictionary
#                             content = chunk_data.get("content", "")
#                         else:
#                             # Fallback for other formats
#                             content = str(chunk_data) if chunk_data else ""
                        
#                         if content:
#                             ai_response += content
#                             yield content
                
#                 # Save updated chat history to MongoDB for context handling
#                 updated_chat_history = chat_history + [
#                     {"role": "user", "content": user_query},
#                     {"role": "assistant", "content": ai_response}
#                 ]
#                 save_chat_history(user_id, updated_chat_history)
                
#                 # Also save a separate entry for UI display purposes
#                 from datetime import datetime
#                 new_chat_entry = {
#                     "user_id": user_id,
#                     "query": user_query,
#                     "response": ai_response,
#                     "timestamp": datetime.utcnow().isoformat()
#                 }
#                 mongo_db["chat_display_history"].insert_one(new_chat_entry)

#             except Exception as stream_error:
#                 print(f"Streaming error: {stream_error}")
#                 error_msg = f"Error during streaming: {str(stream_error)}"
#                 yield error_msg

#         return StreamingResponse(token_generator(), media_type="text/event-stream")

#     except Exception as e:
#         print(f"Ask route error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(data: AskModel):
    try:
        if not data.input:
            raise HTTPException(status_code=400, detail="Input query is required")
        if not data.user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        user_id = str(data.user_id)
        user_query = data.input

        # Get previous conversation from MongoDB.
        chat_history = data.chat_history or get_chat_history(user_id)
        formatted_chat_history = [
            {"role": entry.get("role", ""), "content": entry.get("content", "")}
            for entry in chat_history
        ]

        agent_executor = generate_agent_executor()

        async def token_generator():
            ai_response = ""
            try:
                async for event in agent_executor.astream_events(
                    {"input": user_query, "chat_history": formatted_chat_history},
                    version="v1"
                ):
                    if event.get("event") == "on_chat_model_stream":
                        chunk_data = event.get("data", {}).get("chunk", {})
                        if hasattr(chunk_data, "content"):
                            content = chunk_data.content
                        elif isinstance(chunk_data, dict):
                            content = chunk_data.get("content", "")
                        else:
                            content = str(chunk_data) if chunk_data else ""
                        if content:
                            ai_response += content
                            yield content

                # Append the new messages to the conversation.
                updated_chat_history = chat_history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": ai_response}
                ]
                save_chat_history(user_id, updated_chat_history)
                
            except Exception as stream_error:
                print(f"Streaming error: {stream_error}")
                yield f"Error during streaming: {str(stream_error)}"
        
        return StreamingResponse(token_generator(), media_type="text/event-stream")
    except Exception as e:
        print(f"Ask route error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# Ensure these helper functions are defined
def get_chat_history(user_id: str) -> List[Dict[str, str]]:
    """
    Retrieve chat history for a specific user from MongoDB
    """
    try:
        doc = collection.find_one({"user_id": user_id})
        return doc.get("chat_history", []) if doc else []
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def save_chat_history(user_id: str, chat_history: List[Dict[str, str]]):
    """
    Save chat history for a specific user in MongoDB
    """
    try:
        collection.update_one(
            {"user_id": user_id},
            {"$set": {"chat_history": chat_history}},
            upsert=True
        )
    except Exception as e:
        print(f"Error saving chat history: {e}")


class SaveDashboardModel(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    dashboard_name: Optional[str] = None
    figures: List[str] = Field(..., description="List of figure HTML snippets")

@app.post("/save_dashboard")
def save_dashboard(payload: SaveDashboardModel):
    """
    Save a dashboard for a specific user
    """
    try:
        # Validate user_id
        if not payload.user_id:
            raise HTTPException(status_code=400, detail="User ID is required")

        # Generate unique dashboard filename
        dashboard_name = (
            payload.dashboard_name or 
            f"dashboard_{uuid.uuid4().hex[:8]}.html"
        )
        
        # Combine figures into HTML
        html_parts = [
            "<html>",
            "<head>",
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
            "<style> body { margin: 0; padding: 0; } </style>",
            "</head>",
            "<body>"
        ]
        
        # Add figures to HTML
        html_parts.extend(payload.figures)
        
        html_parts.append("</body></html>")
        html_str = "\n".join(html_parts)

        # For Vercel: Store in MongoDB instead of file system
        if mongo_client and mongo_db:
            dashboard_collection = mongo_db.get_collection("dashboards")
            dashboard_data = {
                "user_id": payload.user_id,
                "dashboard_name": dashboard_name,
                "html_content": html_str,
                "created_at": datetime.datetime.utcnow()
            }
            result = dashboard_collection.insert_one(dashboard_data)
            
            return {
                "status": "success", 
                "dashboard_name": dashboard_name,
                "dashboard_id": str(result.inserted_id)
            }
        
        # Fallback to file system if MongoDB is not available (local dev)
        else:
            # Create user-specific dashboard folder if it doesn't exist
            user_dashboard_folder = os.path.join(DASHBOARD_FOLDER, payload.user_id)
            os.makedirs(user_dashboard_folder, exist_ok=True)
            
            file_path = os.path.join(user_dashboard_folder, dashboard_name)
            
            # Save the dashboard HTML
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            
            return {
                "status": "success", 
                "dashboard_name": dashboard_name, 
                "file_path": file_path
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/get_user_dashboards/{user_id}")
def get_user_dashboards(user_id: str):
    """
    Retrieve all dashboards for a specific user
    """
    # For Vercel: Get dashboards from MongoDB
    if mongo_client and mongo_db:
        try:
            dashboard_collection = mongo_db.get_collection("dashboards")
            dashboards = list(dashboard_collection.find(
                {"user_id": user_id}, 
                {"dashboard_name": 1, "created_at": 1}
            ))
            
            # Convert ObjectId to string for JSON serialization
            for dash in dashboards:
                if "_id" in dash:
                    dash["_id"] = str(dash["_id"])
                if "created_at" in dash:
                    dash["created_at"] = dash["created_at"].isoformat()
            
            return {"dashboards": dashboards}
        except Exception as e:
            print(f"Error retrieving dashboards from MongoDB: {e}")
            return {"dashboards": [], "error": str(e)}
    
    # Fallback to file system (local dev)
    user_dashboard_folder = os.path.join(DASHBOARD_FOLDER, user_id)
    
    if not os.path.exists(user_dashboard_folder):
        return {"dashboards": []}
    
    dashboards = [f for f in os.listdir(user_dashboard_folder) if f.endswith(".html")]
    return {"dashboards": dashboards}

@app.get("/load_user_dashboard/{user_id}/{dashboard_name}")
def load_user_dashboard(user_id: str, dashboard_name: str):
    """
    Load a specific dashboard for a user
    """
    # For Vercel: Get dashboard from MongoDB
    if mongo_client and mongo_db:
        try:
            dashboard_collection = mongo_db.get_collection("dashboards")
            dashboard = dashboard_collection.find_one({
                "user_id": user_id,
                "dashboard_name": dashboard_name
            })
            
            if dashboard and "html_content" in dashboard:
                # Return HTML content directly
                from fastapi.responses import HTMLResponse
                return HTMLResponse(content=dashboard["html_content"], media_type="text/html")
            else:
                raise HTTPException(status_code=404, detail="Dashboard not found")
        except Exception as e:
            print(f"Error retrieving dashboard from MongoDB: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Fallback to file system (local dev)
    dashboard_path = os.path.join(DASHBOARD_FOLDER, user_id, dashboard_name)
    
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    
    raise HTTPException(status_code=404, detail="Dashboard not found")
