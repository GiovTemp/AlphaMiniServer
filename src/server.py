import sys

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import os
import shutil
from tempfile import mkdtemp
import numpy as np
sys.path.append('src')
from ar_vl_preditction import process_and_analyze_aus
from landmarks_dection import detect_landmarks, landmarks_combination_df
from aus import handle_au_and_emotions, predict_emotion

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni l'auth_id accettato dal file .env
ACCEPTED_AUTH_ID = os.getenv("ACCEPTED_AUTH_ID")

app = FastAPI()

# Declare models globally and initialize them as None
au_pred_model = load_model('models/au_pred_model.h5')
arousal_valence_pred_model = load_model('models/arousal_valence_pred_model.h5')

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


@app.post("/upload-image")
async def upload_image(auth_id: str = Form(...), file: UploadFile = File(...)):
    # Verifica l'auth_id qui
    if auth_id != ACCEPTED_AUTH_ID:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Crea una directory temporanea
    temp_dir = mkdtemp()
    image_path = os.path.join(temp_dir, file.filename)

    # Scrivi il file nell'area temporanea
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    landmarks = detect_landmarks(image_path)

    if len(landmarks) > 0:
        landmarks_df = landmarks_combination_df(landmarks)
        prediction = au_pred_model.predict(landmarks_df)
        emotion = predict_emotion(prediction[0])
        au_values = np.array(list(prediction[0]))

        handle_au_and_emotions(au_values, emotion)

        process_and_analyze_aus(arousal_valence_pred_model)

    # Restituisci informazioni sul file ricevuto
    return {"esito": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
