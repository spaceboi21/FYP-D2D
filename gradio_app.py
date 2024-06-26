import gradio as gr
import os
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from PIL import Image
from io import BytesIO

# Set up environment variable for API key
os.environ['GOOGLE_API_KEY'] = "GEMINI_API_KEY"  # replace with your actual key

# Initialize chat history
chat_history = []

# Define generation config
generation_config = genai.GenerationConfig(
    stop_sequences=None,
    temperature=0.4,  # Updated temperature
    top_p=1.0,        # Updated top_p
    top_k=32,         # Updated top_k
    candidate_count=1,
    max_output_tokens=4097  # Updated token limit
)

def llm_function(messages, use_vision_model):
    model_name = "gemini-pro-vision" if use_vision_model else "gemini-pro"
    llm = ChatGoogleGenerativeAI(model=model_name, generation_config=generation_config)
    response = llm.invoke([HumanMessage(content=messages)])
    disp = response.content

    # Storing the Assistant Message
    chat_history.append({
        "role": "assistant",
        "content": disp
    })
    return disp

def process_inputs(query, uploaded_file):
    message_content = []
    use_vision_model = False

    if query:
        full_user_message = system_prompt + query
        message_content.append({
            "type": "text",
            "text": full_user_message
        })

    if uploaded_file:
        use_vision_model = True
        buffered = BytesIO()
        uploaded_file.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{base64_image}"
        message_content.append({
            "type": "image_url",
            "image_url": data_url
        })
    
    response = llm_function(message_content, use_vision_model)
    return response

def update_chatbox(query, uploaded_file):
    response = process_inputs(query, uploaded_file)
    chat = []
    for message in chat_history:
        role, content = message["role"], message["content"]
        if role == "assistant" and isinstance(content, list):
            for item in content:
                if item["type"] == "image_url":
                    image_data = base64.b64decode(item["image_url"].split(",")[1])
                    image = Image.open(BytesIO(image_data))
                    chat.append((role, gr.Image.update(value=image)))
                else:
                    chat.append((role, content))
        else:
            chat.append((role, content))
    return chat

# Define the Gradio interface
with gr.Blocks(css="footer{display:none !important}") as demo:
    gr.Markdown("# MUSA Assistant")

    with gr.Row():
        with gr.Column():
            chat_output = gr.Chatbot(label="Chat History")
            uploaded_file = gr.Image(label="Upload image", sources="upload", type="pil")
            query = gr.Textbox(label="Hello! How may I help you today?")
            submit_btn = gr.Button("Submit")

    # System prompt to be prepended
    system_prompt = "You are a medical help assistant and have to answer questions regarding the images uploaded. You have to talk about the disease and ways to reduce the symptoms and pain. "
    
    def submit_fn(query, uploaded_file):
        chat_history = update_chatbox(query, uploaded_file)
        return chat_history

    submit_btn.click(submit_fn, inputs=[query, uploaded_file], outputs=[chat_output])

# Launch the Gradio app
demo.launch(share=True)
