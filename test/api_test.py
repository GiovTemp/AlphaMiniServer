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