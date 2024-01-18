from fastapi.testclient import TestClient
import sys
sys.path.append('src')
from server import app #errore falso positivo
import os
from dotenv import load_dotenv

client = TestClient(app)

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni l'auth_id accettato dal file .env
ACCEPTED_AUTH_ID = os.getenv("ACCEPTED_AUTH_ID")

def test_agent_endpoint_success():
    response = client.post("/agent", json={
        "auth_id": ACCEPTED_AUTH_ID,  # Inserisci l'auth_id accettato qui
        "robot_id": "123",
        "text": "Hello, robot!"
    })
    assert response.status_code == 200
    assert response.json() == {
        "action": "example_action",
        "answer": "This is a response to: Hello, robot!"
    }

def test_agent_endpoint_unauthorized():
    response = client.post("/agent", json={
        "auth_id": "wrong_auth_id",
        "robot_id": "123",
        "text": "Hello, robot!"
    })
    assert response.status_code == 401

def test_agent_endpoint_missing_fields():
    response = client.post("/agent", json={
        "auth_id": ACCEPTED_AUTH_ID,
        # mancano 'robot_id' e 'text'
    })
    assert response.status_code == 422
