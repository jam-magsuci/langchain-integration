from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

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

# Initialize HuggingFace model
def init_model():
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        return llm
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        llm = init_model()
        if not llm:
            raise HTTPException(status_code=500, detail="Failed to initialize language model")
        
        # Generate response
        response = llm.predict(request.question)
        
        return {
            "response": response,
            "conversation_id": request.conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)