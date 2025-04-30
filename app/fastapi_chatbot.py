from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional

import os
from dotenv import load_dotenv
import base64
import pandas as pd
from io import StringIO

from pymongo import MongoClient

# Pinecone / LangChain
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
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
import json
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import dash
from dash import html, dcc
import plotly.graph_objs as go
import plotly.io as pio

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # or store in .env

DASHBOARD_FOLDER = "saved_dashboards"  # Folder to save user dashboards

# Define the system prompt for the AI assistant
SYSTEM_PROMPT = """You are a helpful data analysis assistant with access to a retriever tool that can query a database of CSV data.

When you receive data from the retriever tool, carefully examine the actual values in the data and use them in your response.
DO NOT use placeholder text like [department name] or [sales figure] - instead, extract the specific values from the returned data.

For example:
- If you see PRODUCTLINE: Ships, use "Ships" as the product line
- If you see SALES: 3515.7, use "3515.7" as the sales value
- If you see TOTAL: 250.35, use "250.35" as the total value

Format numbers appropriately with proper currency symbols or units where applicable.
Always base your answers on the actual data returned by the tool, not on general knowledge.

Remember that each user has their own separate data. Do not reference other users' data in your responses.
If the retriever returns no data or errors, politely inform the user that their data may not have been uploaded correctly.

For data visualization suggestions:
1. Be specific about which columns would work well in charts
2. Recommend appropriate chart types for the data structure
3. Explain why your visualization recommendations would be insightful

Always maintain a helpful, professional tone and focus on providing accurate insights based on the user's specific data.
"""

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
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "fyp"  # 768 dims, metric=cosine
index_list = pc.list_indexes()
if index_name not in [idx.name for idx in index_list]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1
        )
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
    user_id: str
    csv_base64: str

class AskModel(BaseModel):
    input: str
    user_id: str = Field(..., description="Unique identifier for the user")
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    namespace: Optional[str] = None  # Add namespace parameter

# ---------------------------------------------------
# 3) Pinecone VectorStore + Retriever
# ---------------------------------------------------
def load_db(namespace=None):
    """
    Load the Pinecone vector store with the specified namespace.
    If no namespace is provided, use 'default'.
    """
    if namespace is None:
        namespace = "default"
        
    store = PineconeVectorStore(
        index_name=index_name,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY),
        namespace=namespace
    )
    return store

def create_retriever(namespace=None):
    """
    Create a retriever for the specified namespace.
    Implements better error handling and validation.
    """
    try:
        # Set a default namespace if none provided
        if namespace is None:
            namespace = "default"
            print(f"WARNING: No namespace provided to create_retriever, using '{namespace}'")
        
        # Create embeddings with error handling
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f"ERROR: Failed to create embeddings: {e}")
            raise ValueError(f"Embeddings initialization failed: {str(e)}")
        
        # Create vector database with error handling
        try:
            vector_db = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                namespace=namespace
            )
        except Exception as e:
            print(f"ERROR: Failed to load vector database for namespace '{namespace}': {e}")
            raise ValueError(f"Vector database initialization failed: {str(e)}")
        
        # Create retriever with namespace filtering
        try:
            retriever = vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 10,
                    "score_threshold": 0.5,
                    "namespace": namespace  # Use the correct namespace parameter
                }
            )
        except Exception as e:
            print(f"ERROR: Failed to create retriever: {e}")
            raise ValueError(f"Retriever creation failed: {str(e)}")
        
        # Create tool with better description
        try:
            tool = create_retriever_tool(
                retriever,
                name="csv_data_retriever",
                description=f"Retriever tool to query CSV data from namespace '{namespace}'. Use this to answer questions about the user's data. Always check the namespace to ensure you're accessing the correct user's data."
            )
            print(f"DEBUG: Successfully created retriever tool for namespace '{namespace}'")
            return tool
        except Exception as e:
            print(f"ERROR: Failed to create retriever tool: {e}")
            raise ValueError(f"Retriever tool creation failed: {str(e)}")
            
    except Exception as e:
        print(f"CRITICAL ERROR in create_retriever: {e}")
        # Return a dummy tool that will inform the user about the error
        def dummy_retriever(*args, **kwargs):
            return [f"Error retrieving data: {str(e)}. Please try uploading your data again."]
        
        return create_retriever_tool(
            dummy_retriever, 
            name="error_retriever",
            description="This retriever encountered an error. Ask the user to upload their data again."
        )

# ---------------------------------------------------
# 4) Agent Executor
# ---------------------------------------------------
def generate_agent_executor(namespace=None):
    """
    Generate an agent executor with error handling.
    """
    try:
        # Get the retriever tool for the namespace
        retriever_tool = create_retriever(namespace)
        
        # Create the LLM with error handling
        try:
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                streaming=True,
                openai_api_key=OPENAI_API_KEY
            )
        except Exception as e:
            print(f"ERROR: Failed to create LLM: {e}")
            raise ValueError(f"LLM initialization failed: {str(e)}")
        
        # Create prompt with error handling
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        except Exception as e:
            print(f"ERROR: Failed to create prompt: {e}")
            raise ValueError(f"Prompt creation failed: {str(e)}")
        
        # Create agent with error handling
        try:
            agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
        except Exception as e:
            print(f"ERROR: Failed to create agent: {e}")
            raise ValueError(f"Agent creation failed: {str(e)}")
        
        # Create agent executor with error handling
        try:
            return AgentExecutor(
                agent=agent,
                tools=[retriever_tool],
                verbose=True,
                return_intermediate_steps=True
            )
        except Exception as e:
            print(f"ERROR: Failed to create agent executor: {e}")
            raise ValueError(f"Agent executor creation failed: {str(e)}")
            
    except Exception as e:
        print(f"CRITICAL ERROR in generate_agent_executor: {e}")
        # Create a simple AgentExecutor that will just return an error message
        def error_function(*args, **kwargs):
            return {"output": f"Error initializing the AI assistant: {str(e)}. Please contact support or try again later."}
            
        # This is a minimal agent executor that will just return the error
        class SimpleErrorExecutor:
            def __init__(self, error_msg):
                self.error_msg = error_msg
                
            async def astream_events(self, *args, **kwargs):
                yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": self.error_msg}}}
                
        return SimpleErrorExecutor(f"Error initializing AI assistant: {str(e)}. Please try again later.")

async def stream_agent_response(agent_executor, user_query, user_id, messages):
    """
    Stream the agent response with proper error handling and chat history updates.
    """
    ai_response = ""
    try:
        print(f"DEBUG: Starting token generation for user {user_id}")
        async for event in agent_executor.astream_events(
            {"input": user_query, "chat_history": messages},
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
                    print(f"DEBUG: Generated token: {content[:20]}..." if len(content) > 20 else f"DEBUG: Generated token: {content}")
                    yield content

        # Append the new messages to the conversation.
        try:
            updated_chat_history = messages + [
                {"role": "assistant", "content": ai_response}
            ]
            save_chat_history(user_id, updated_chat_history)
            print(f"DEBUG: Chat history updated with new messages for user {user_id}")
        except Exception as history_error:
            print(f"DEBUG: Error updating chat history: {history_error}")
            # Non-critical error, don't interrupt the stream
            
    except Exception as stream_error:
        error_msg = f"Error during processing: {str(stream_error)}"
        print(f"DEBUG: Streaming error: {error_msg}")
        yield error_msg

# ---------------------------------------------------
# 5) /upload_csv Endpoint -> upsert CSV to Pinecone
# ---------------------------------------------------
@app.post("/upload_csv")
def upload_csv(payload: UploadCSVModel):
    print(f"DEBUG: upload_csv endpoint called with session_id={payload.session_id}, user_id={payload.user_id}, csv_data_length={len(payload.csv_base64) if payload.csv_base64 else 0}")
    try:
        # Make sure we have a valid user_id
        if not payload.user_id:
            if payload.session_id:
                payload.user_id = payload.session_id
            else:
                raise ValueError("Either user_id or session_id must be provided")
                
        # Create a unique namespace for the user
        user_namespace = f"user_{payload.user_id}"
        
        # First check if the user already has data - if so, delete it
        try:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            if user_namespace in stats.namespaces:
                print(f"DEBUG: Deleting existing namespace {user_namespace} for user {payload.user_id}")
                index.delete(namespace=user_namespace, delete_all=True)
                print(f"DEBUG: Successfully deleted namespace {user_namespace}")
        except Exception as e:
            print(f"DEBUG: Error checking/clearing user namespace: {e}")
        
        # Process the CSV data
        raw_b64 = payload.csv_base64
        decoded = base64.b64decode(raw_b64)
        csv_str = decoded.decode("utf-8", errors="ignore")
        print(f"DEBUG: Decoded CSV data: {csv_str[:100]}...")
        
        try:
            df = pd.read_csv(StringIO(csv_str))
        except Exception as csv_err:
            print(f"DEBUG: Error parsing CSV: {csv_err}")
            raise ValueError(f"Failed to parse CSV: {str(csv_err)}")
            
        if df.empty:
            raise ValueError("CSV is empty or invalid.")

        print(f"DEBUG: CSV loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        
        # Convert CSV to text in smaller chunks to avoid memory issues
        chunk_size = 100  # Process 100 rows at a time
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        processed_chunks = 0
        
        store = PineconeVectorStore(
            index_name=index_name,
            embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY),
            namespace=user_namespace
        )
        
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            chunk_text = chunk_df.to_csv(index=False)
            
            # Split text into smaller chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Reduced chunk size
                chunk_overlap=50,  # Small overlap to maintain context
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_text(chunk_text)
            docs = [Document(page_content=ch) for ch in chunks]
            
            # Add chunks to Pinecone with retries
            max_retries = 3
            for retry in range(max_retries):
                try:
                    store.add_documents(docs)
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        print(f"DEBUG: Failed to add chunk {processed_chunks + 1}/{total_chunks} after {max_retries} retries: {e}")
                        raise
                    print(f"DEBUG: Retry {retry + 1} for chunk {processed_chunks + 1}")
                    time.sleep(2 ** retry)  # Exponential backoff
            
            processed_chunks += 1
            print(f"DEBUG: Processed chunk {processed_chunks}/{total_chunks}")
        
        # Update user info in MongoDB
        if mongo_client is not None and mongo_db is not None:
            try:
                user_collection = mongo_db.get_collection("users")
                user_collection.update_one(
                    {"user_id": payload.user_id},
                    {"$set": {
                        "has_uploaded_csv": True,
                        "last_upload": datetime.datetime.utcnow(),
                        "csv_rows": len(df),
                        "csv_columns": len(df.columns)
                    }},
                    upsert=True
                )
                print(f"DEBUG: Updated user upload info in MongoDB for user {payload.user_id}")
            except Exception as mongo_err:
                print(f"DEBUG: Error updating MongoDB user info: {mongo_err}")
        
        return {"status": "success", "rows_ingested": len(df), "namespace": user_namespace}
        
    except ValueError as ve:
        print(f"DEBUG: Validation error in upload_csv: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"DEBUG: Unexpected error in upload_csv: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ---------------------------------------------------
# 6) /ask Endpoint -> streaming response
# ---------------------------------------------------
@app.post("/ask")
async def ask_question(data: AskModel):
    print(f"DEBUG: ask endpoint called with user_id={data.user_id}, input={data.input}")
    try:
        if not data.input:
            raise HTTPException(status_code=400, detail="Input query is required")
        
        user_id = None
        # First try to get user_id from the request
        if data.user_id:
            user_id = str(data.user_id)
        # If not available, fall back to session_id
        elif data.session_id:
            user_id = str(data.session_id)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User identification (user_id or session_id) is required")
        
        # Create a user-specific namespace for Pinecone retrieval
        # If namespace is provided in the payload, use it instead of creating a new one
        user_namespace = data.namespace if data.namespace else f"user_{user_id}"
        
        print(f"DEBUG: Using namespace {user_namespace} for user {user_id}")
        user_query = data.input

        # Get previous conversation from MongoDB
        chat_history = data.chat_history or get_chat_history(user_id)
        print(f"DEBUG: Retrieved chat history with {len(chat_history)} entries")
        formatted_chat_history = [
            {"role": entry.get("role", ""), "content": entry.get("content", "")}
            for entry in chat_history
        ]

        # Check if the user has data in their namespace
        try:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            
            if user_namespace not in stats.namespaces:
                print(f"DEBUG: User {user_id} has no data in namespace {user_namespace}")
                return StreamingResponse(
                    stream_error_response("No data found for your account. Please upload a CSV file first."),
                    media_type="text/event-stream"
                )
                
            vector_count = stats.namespaces.get(user_namespace, {}).get("vector_count", 0)
            print(f"DEBUG: Namespace {user_namespace} has {vector_count} vectors")
                
            if vector_count == 0:
                print(f"DEBUG: User {user_id} has empty namespace {user_namespace}")
                return StreamingResponse(
                    stream_error_response("Your data appears to be empty. Please upload a valid CSV file."),
                    media_type="text/event-stream"
                )
        except Exception as e:
            print(f"DEBUG: Error checking user namespace: {e}")
            # Continue anyway, the query might just fail later
        
        # Pass user_namespace to generate_agent_executor for user-specific data retrieval
        agent_executor = generate_agent_executor(user_namespace)
        print(f"DEBUG: Created agent executor for user {user_id} with namespace {user_namespace}")
        
        # Create message list with system prompt and chat history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if formatted_chat_history:
            messages.extend(formatted_chat_history)
        messages.append({"role": "user", "content": user_query})
        
        # Execute the agent in a background task and stream the response
        return StreamingResponse(
            stream_agent_response(agent_executor, user_query, user_id, messages),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"DEBUG: Error in ask_question: {e}")
        # Stream the error back to the client
        return StreamingResponse(
            stream_error_response(f"Error processing your request: {str(e)}"),
            media_type="text/event-stream"
        )

async def stream_error_response(error_message):
    """Stream an error message back to the client"""
    yield error_message

class SaveDashboardModel(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    dashboard_name: Optional[str] = None
    figures: List[Dict] = Field(..., description="List of Plotly figure dictionaries")

@app.post("/save_dashboard")
async def save_dashboard(request: Request, background_tasks: BackgroundTasks):
    """
    Save the dashboard data to a file.
    """
    try:
        data = await request.json()
        logger.info("Starting save_dashboard process")
        
        # Verify that we have the required fields
        if not all(k in data for k in ['user_id', 'figures', 'dashboard_name']):
            logger.error(f"Missing required fields in save_dashboard request: {data.keys()}")
            return {"success": False, "error": "Missing required fields"}
        
        user_id = data.get('user_id')
        figures = data.get('figures', [])
        dashboard_name = data.get('dashboard_name', f"Dashboard-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        logger.info(f"Saving dashboard for user {user_id} with {len(figures)} figures")
        
        # Create user directory if it doesn't exist
        base_dir = os.path.join(os.getcwd(), 'app', 'dashboards')
        user_dir = os.path.join(base_dir, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"Created/verified user directory: {user_dir}")
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_name = ''.join(c for c in dashboard_name if c.isalnum() or c in [' ', '_', '-']).strip().replace(' ', '_')
        filename = f"{safe_name}_{timestamp}.html"
        filepath = os.path.join(user_dir, filename)
        logger.info(f"Generated filepath: {filepath}")
        
        # Create a new Dash app
        app = dash.Dash(__name__, assets_folder='assets')
        
        # Create layout with all figures
        app.layout = html.Div([
            html.H1(dashboard_name, className="mb-4"),
            html.Div([
                html.Div([
                    dcc.Graph(figure=go.Figure(fig), className="mb-4")
                ]) for fig in figures
            ])
        ], className="container")
        
        # Generate and save the HTML
        app.index_string = """
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        """
        
        # Save the dashboard HTML
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(app.index_string)
        logger.info(f"Saved dashboard HTML to {filepath}")
        
        # Save figure data for later reconstruction
        fig_filepath = os.path.join(user_dir, f"{safe_name}_{timestamp}_data.json")
        with open(fig_filepath, 'w', encoding='utf-8') as f:
            json.dump(figures, f, cls=CustomJSONEncoder)
        logger.info(f"Saved figure data to {fig_filepath}")
        
        # Add dashboard info to SQLite database via SQLAlchemy
        background_tasks.add_task(
            add_dashboard_to_db,
            user_id=int(user_id),
            dashboard_name=dashboard_name,
            file_path=filepath,
            fig_data_path=fig_filepath
        )
        logger.info("Added dashboard to background tasks for database update")
        
        return {
            "success": True, 
            "message": "Dashboard saved successfully",
            "filepath": filepath
        }
    except Exception as e:
        logger.error(f"Error saving dashboard: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

# Add this class for JSON serialization of plotly figures
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, 'to_json'):
            return obj.to_json()
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def add_dashboard_to_db(user_id, dashboard_name, file_path, fig_data_path=None):
    """
    Add dashboard information to the database.
    """
    try:
        engine = create_engine(SQLALCHEMY_DATABASE_URI)
        Base = declarative_base()
        
        class Dashboard(Base):
            __tablename__ = 'dashboards'
            id = Column(Integer, primary_key=True)
            user_id = Column(Integer, nullable=False)
            dashboard_name = Column(String(100), nullable=True)
            file_path = Column(String(500), nullable=False)
            fig_data_path = Column(String(500), nullable=True)
            created_at = Column(DateTime, default=datetime.utcnow)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Add new dashboard
        new_dashboard = Dashboard(
            user_id=user_id,
            dashboard_name=dashboard_name,
            file_path=file_path,
            fig_data_path=fig_data_path
        )
        
        session.add(new_dashboard)
        session.commit()
        logging.info(f"Added dashboard to database: {dashboard_name} for user {user_id}")
        
        # Close session
        session.close()
        return True
    except Exception as e:
        logging.error(f"Error adding dashboard to database: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

@app.get("/get_user_dashboards/{user_id}")
def get_user_dashboards(user_id: str):
    """
    Retrieve all dashboards for a specific user
    """
    # For Vercel: Get dashboards from MongoDB
    if mongo_client is not None and mongo_db is not None:
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

@app.get("/load_user_dashboard/{user_id}/{filename}")
async def load_user_dashboard(user_id: str, filename: str):
    """
    Load a previously saved dashboard by filename
    """
    try:
        file_path = os.path.join(DASHBOARD_FOLDER, user_id, filename)
        
        if not os.path.exists(file_path):
            # Try looking for data JSON if HTML file is not found
            if filename.endswith('.html'):
                data_filename = filename.replace('.html', '_data.json')
                data_path = os.path.join(DASHBOARD_FOLDER, user_id, data_filename)
                
                if os.path.exists(data_path):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        fig_data = json.load(f)
                    
                    # Rebuild dashboard from data
                    reconstructed_html = rebuild_dashboard_from_data(fig_data)
                    
                    # Save the reconstructed HTML
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(reconstructed_html)
                    
                    logging.info(f"Reconstructed dashboard from data: {file_path}")
                    return FileResponse(file_path)
            
            return {"error": f"Dashboard file not found: {file_path}"}
        
        return FileResponse(file_path)
    except Exception as e:
        logging.error(f"Error loading dashboard: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}

def rebuild_dashboard_from_data(fig_data):
    """
    Rebuild a dashboard HTML from saved figure data
    """
    try:
        # Create a new Dash app
        app = dash.Dash(__name__, assets_folder='assets')
        
        # Reconstruct figures from the data
        figures = []
        for fig_dict in fig_data:
            try:
                fig = go.Figure(fig_dict)
                figures.append(fig)
            except Exception as e:
                logging.error(f"Error reconstructing figure: {str(e)}")
                continue
        
        if not figures:
            logging.error("No valid figures could be reconstructed")
            return f"<html><body><h1>Error Rebuilding Dashboard</h1><p>No valid figures could be reconstructed</p></body></html>"
        
        # Create a layout with the reconstructed figures
        app.layout = html.Div([
            html.H1("Reconstructed Dashboard", className="mb-4"),
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig, className="mb-4")
                ]) for fig in figures
            ])
        ], className="container")
        
        # Generate HTML
        dashboard_html = app.index_string
        
        return dashboard_html
    
    except Exception as e:
        logging.error(f"Error rebuilding dashboard: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return f"<html><body><h1>Error</h1><p>Error rebuilding dashboard: {str(e)}</p></body></html>"

@app.get("/view_dashboard/{dashboard_id}")
async def view_dashboard(dashboard_id: str, request: Request):
    """
    Retrieve and display a saved dashboard
    """
    try:
        logger.info(f"Starting view_dashboard for dashboard_id: {dashboard_id}")
        
        # Get user_id from request headers or query params
        user_id = request.headers.get('X-User-ID') or request.query_params.get('user_id')
        if not user_id:
            logger.error("No user_id provided")
            return HTMLResponse(content="<h1>Error</h1><p>User ID is required</p>", status_code=400)
            
        logger.info(f"Current user: {user_id}")
        
        # First verify the database structure
        try:
            # Check if dashboard table exists
            engine = create_engine(SQLALCHEMY_DATABASE_URI)
            with engine.connect() as connection:
                result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='dashboard'"))
                table_exists = result.fetchone()
                logger.info(f"Dashboard table exists: {bool(table_exists)}")
                
                if not table_exists:
                    logger.error("Dashboard table does not exist")
                    return HTMLResponse(content="""
                    <h2>Database Error</h2>
                    <p>The dashboard table does not exist in the database.</p>
                    <p>Please try saving a dashboard first to create the table.</p>
                    """, status_code=500)
                
                # Get count of dashboards
                count_result = connection.execute(text("SELECT COUNT(*) FROM dashboard"))
                dashboard_count = count_result.scalar()
                logger.info(f"Total dashboards in database: {dashboard_count}")
                
                # Get all dashboards
                all_dashboards = connection.execute(text("SELECT * FROM dashboard")).fetchall()
                logger.info(f"Found {len(all_dashboards)} dashboards in database")
                
                debug_info = f"""
                <h2>Database Status</h2>
                <p>Current User ID: {user_id}</p>
                <p>Requested Dashboard ID: {dashboard_id}</p>
                <p>Total Dashboards in Database: {dashboard_count}</p>
                
                <h3>All Dashboards:</h3>
                <table border='1'>
                    <tr>
                        <th>ID</th>
                        <th>User ID</th>
                        <th>Name</th>
                        <th>File Path</th>
                    </tr>
                """
                
                for dash in all_dashboards:
                    debug_info += f"""
                    <tr>
                        <td>{dash[0]}</td>
                        <td>{dash[1]}</td>
                        <td>{dash[3]}</td>
                        <td>{dash[2]}</td>
                    </tr>
                    """
                    logger.info(f"Dashboard found - ID: {dash[0]}, User ID: {dash[1]}, Path: {dash[2]}")
                
                debug_info += "</table>"
                
                if dashboard_count == 0:
                    logger.warning("No dashboards found in database")
                    return HTMLResponse(content=debug_info + "<p>No dashboards found in the database. Please save a dashboard first.</p>", status_code=404)
                    
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            import traceback
            logger.error(traceback.format_exc())
            return HTMLResponse(content=f"""
            <h2>Database Error</h2>
            <p>Error accessing database: {str(db_error)}</p>
            <p>Please check if the database is properly initialized.</p>
            """, status_code=500)
        
        # Get the dashboard from the database with user check
        try:
            Session = sessionmaker(bind=engine)
            session = Session()
            dashboard = session.query(Dashboard).filter_by(id=int(dashboard_id), user_id=int(user_id)).first()
            session.close()
            
            if not dashboard:
                logger.error(f"Dashboard {dashboard_id} not found for user {user_id}")
                return HTMLResponse(content=debug_info + f"<p>Dashboard {dashboard_id} not found for user {user_id}</p>", status_code=404)
                
            logger.info(f"Found dashboard with path: {dashboard.file_path}")
            
            # Check if the file exists
            file_path = dashboard.file_path
            logger.info(f"Checking file existence at: {file_path}")
            
            if not os.path.exists(file_path):
                # Try to find the file in the user's dashboard directory
                base_dir = os.path.join(os.getcwd(), 'app', 'dashboards')
                user_dir = os.path.join(base_dir, str(user_id))
                possible_file = os.path.join(user_dir, os.path.basename(file_path))
                
                logger.info(f"Original file path not found: {file_path}")
                logger.info(f"Checking alternative path: {possible_file}")
                
                if os.path.exists(possible_file):
                    file_path = possible_file
                    logger.info(f"Found file at alternative path: {file_path}")
                else:
                    logger.error(f"File not found at either {file_path} or {possible_file}")
                    return HTMLResponse(content=debug_info + f"<p>Dashboard file not found at either location:</p><ul><li>{file_path}</li><li>{possible_file}</li></ul>", status_code=404)
            
            # Serve the file
            logger.info(f"Serving file: {file_path}")
            return FileResponse(
                file_path,
                media_type='text/html',
                filename=os.path.basename(file_path)
            )
            
        except Exception as e:
            logger.error(f"Error serving dashboard {dashboard_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return HTMLResponse(content=f"Error: {str(e)}", status_code=500)
            
    except Exception as e:
        logger.error(f"Unexpected error in view_dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return HTMLResponse(content=f"Unexpected error: {str(e)}", status_code=500)

@app.get("/health")
def health_check():
    """
    Simple health check endpoint to test if the server is running
    """
    return {"status": "ok", "message": "FastAPI server is running"}
