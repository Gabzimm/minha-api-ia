from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Liberar CORS para seu site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, coloque a URL do seu site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY não configurada!")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    # Usar o modelo mais recente
    model = genai.GenerativeModel('gemini-1.5-pro')  # ou 'gemini-1.5-flash' para mais rápido/barato

class ChatRequest(BaseModel):
    mensagem: str
    session_id: str = None
    temperatura: float = 0.7

class ChatResponse(BaseModel):
    resposta: str
    session_id: str
    timestamp: str

@app.get("/")
async def root():
    return {
        "message": "API AI WaveX com Gemini",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "gemini"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Recebida mensagem: {request.mensagem[:50]}...")
        
        # Verificar se a chave está configurada
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY não configurada no ambiente"
            )
        
        # Configurar temperatura (Gemini usa parâmetros diferentes)
        generation_config = {
            "temperature": request.temperatura,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Criar prompt com contexto (opcional)
        prompt = f"""Você é a AI WaveX, um assistente amigável e útil.
        
Usuário: {request.mensagem}
AI WaveX:"""
        
        # Gerar resposta com Gemini
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extrair texto da resposta
        if response.text:
            resposta = response.text
        else:
            resposta = "Desculpe, não consegui gerar uma resposta."
        
        logger.info("Resposta gerada com sucesso")
        
        return ChatResponse(
            resposta=resposta,
            session_id=request.session_id or f"sessao_{datetime.now().timestamp()}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modelos")
async def listar_modelos():
    """Lista os modelos Gemini disponíveis"""
    try:
        modelos = genai.list_models()
        return {
            "modelos": [
                {
                    "name": m.name,
                    "description": m.description,
                    "supported_methods": m.supported_generation_methods
                }
                for m in modelos
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
