from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from typing import Dict, Tuple
import time

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    question: str
    conversation_id: str

# Define system message and context
SYSTEM_MESSAGE = """You are a helpful AI assistant focused on providing clear and accurate information. 
Please be concise and make a bullet point list of the answer and end each bullet point with a <br><br> tag."""

def format_prompt(question: str) -> str:
    return f"""System: {SYSTEM_MESSAGE}<br><br>
User Question: {question}<br><br>
Assistant: Let me help you with that.<br><br>"""

# Initialize HuggingFace model
def init_model():
    try:
        llm = HuggingFaceHub(
            # repo_id="google/flan-t5-base",
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        return llm
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

# Add cache configuration
CACHE_DURATION = 3600  # Cache duration in seconds (1 hour)
response_cache: Dict[str, Tuple[str, float]] = {}  # {question: (response, timestamp)}

def get_cached_response(question: str) -> str | None:
    if question in response_cache:
        response, timestamp = response_cache[question]
        if time.time() - timestamp < CACHE_DURATION:
            return response
        else:
            # Remove expired cache entry
            del response_cache[question]
    return None

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Check cache first
        cached_response = get_cached_response(request.question)
        if cached_response:
            return {
                "response": cached_response,
                "conversation_id": request.conversation_id,
                "cached": True
            }

        llm = init_model()
        if not llm:
            raise HTTPException(status_code=500, detail="Failed to initialize language model")
        
        formatted_prompt = format_prompt(request.question)
        response = llm.predict(formatted_prompt)
        
        # Store in cache
        response_cache[request.question] = (response, time.time())
        
        return {
            "response": response,
            "conversation_id": request.conversation_id,
            "cached": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)