from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from datetime import datetime
import logging
from typing import Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="AI WaveX API",
    description="API da AI WaveX usando Google Gemini",
    version="1.0.0"
)

# Configurar CORS
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
    logger.error("⚠️ GEMINI_API_KEY não configurada!")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    # Usar modelo mais recente
    model = genai.GenerativeModel('gemini-1.5-flash')  # Mais rápido e grátis
    logger.info("✅ Gemini configurado com sucesso!")

# Modelos de dados
class ChatRequest(BaseModel):
    mensagem: str
    session_id: Optional[str] = None
    temperatura: Optional[float] = 0.7

class ChatResponse(BaseModel):
    resposta: str
    session_id: str
    timestamp: str

# Dicionário para histórico (simples)
historico = {}

@app.get("/")
async def root():
    """Página inicial da API"""
    return {
        "message": "🚀 AI WaveX API com Google Gemini",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Verificar saúde da API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "gemini-1.5-flash",
        "gemini_configured": bool(GEMINI_API_KEY)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint para conversar com a AI WaveX
    """
    try:
        logger.info(f"📨 Mensagem recebida: {request.mensagem[:50]}...")
        
        # Verificar se a chave está configurada
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY não configurada. Configure no ambiente do Render."
            )
        
        # Criar ou recuperar sessão
        session_id = request.session_id or f"sessao_{int(datetime.now().timestamp())}"
        
        # Configurar geração
        generation_config = {
            "temperature": request.temperatura,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Criar prompt com contexto
        prompt = f"""Você é a AI WaveX, um assistente virtual amigável, útil e falante de português.

Histórico da conversa:
{historico.get(session_id, 'Início da conversa')}

Usuário: {request.mensagem}
AI WaveX:"""
        
        # Gerar resposta
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extrair resposta
        if response.text:
            resposta = response.text
        else:
            resposta = "Desculpe, não consegui processar sua mensagem. Pode repetir?"
        
        # Salvar no histórico
        if session_id not in historico:
            historico[session_id] = []
        
        historico[session_id].append(f"Usuário: {request.mensagem}")
        historico[session_id].append(f"AI WaveX: {resposta[:50]}...")
        
        # Manter apenas últimas 10 mensagens
        if len(historico[session_id]) > 20:
            historico[session_id] = historico[session_id][-20:]
        
        logger.info(f"✅ Resposta gerada para sessão {session_id}")
        
        return ChatResponse(
            resposta=resposta,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modelos")
async def listar_modelos():
    """Listar modelos Gemini disponíveis"""
    try:
        modelos = genai.list_models()
        return {
            "modelos": [
                {
                    "nome": m.name,
                    "descricao": m.description,
                    "metodos": m.supported_generation_methods
                }
                for m in modelos
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/historico/{session_id}")
async def limpar_historico(session_id: str):
    """Limpar histórico de uma sessão"""
    if session_id in historico:
        del historico[session_id]
        return {"message": f"Histórico da sessão {session_id} removido"}
    return {"message": f"Sessão {session_id} não encontrada"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
