from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni l'auth_id accettato dal file .env
ACCEPTED_AUTH_ID = os.getenv("ACCEPTED_AUTH_ID")

app = FastAPI()

class AgentRequest(BaseModel):
    auth_id: str
    robot_id: str
    text: str

class AgentResponse(BaseModel):
    action: str
    answer: str

@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    # Verifica l'autenticazione
    if request.auth_id != ACCEPTED_AUTH_ID:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Logica per determinare 'action' e 'answer'
    action = "example_action"
    answer = "This is a response to: " + request.text

    return AgentResponse(action=action, answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
