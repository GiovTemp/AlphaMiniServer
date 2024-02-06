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


def test_upload_image():
    # This will give you the absolute path of the directory where the script is located.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, 'images', 'face.jpg')
    auth_id = ACCEPTED_AUTH_ID

    # Simulate a POST request to the endpoint
    # Open the image in binary mode
    with open(image_path, 'rb') as image_file:
        # Simulate a POST request to the endpoint
        response = client.post(
            "/upload-image",
            files={"file": (image_path, image_file, "image/jpeg")},
            data={"auth_id": auth_id}
        )

    # Verify the response
    assert response.status_code == 200
    assert response.json() == {"esito": "ok"}

def test_upload_audio():
    # Questo ti darà il percorso assoluto della directory in cui si trova lo script.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(BASE_DIR, 'audio', 'test_audio.wav')  # Sostituisci con il tuo file audio
    auth_id = ACCEPTED_AUTH_ID

    # Simula una richiesta POST all'endpoint
    # Apri il file audio in modalità binaria
    with open(audio_path, 'rb') as audio_file:
        # Simula una richiesta POST all'endpoint
        response = client.post(
            "/upload-audio",
            files={"file": (audio_path, audio_file, "audio/wav")},
            data={"auth_id": auth_id}
        )

    print(response.json())
    # Verifica la risposta
    assert response.status_code == 200
    assert "testo" in response.json()  # Verifica che il campo 'testo' sia presente nella risposta JSON

    # Verifica che il testo trascritto contenga certe parole chiave o frasi
    expected_keywords = ["Alessio", "Policoro", "registrazione"]
    actual_text = response.json()["testo"]['input']
    for keyword in expected_keywords:
        assert keyword in actual_text, f"Expected '{keyword}' in the transcription, but it was not found."
