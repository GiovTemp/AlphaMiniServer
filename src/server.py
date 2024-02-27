import sys

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
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

from vosk import Model, KaldiRecognizer
import json
import tempfile
import openai
import time
from pydub import AudioSegment

model_path = "vosk-model-it-0.22"
model = Model(model_path)

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni l'auth_id accettato dal file .env
ACCEPTED_AUTH_ID = os.getenv("ACCEPTED_AUTH_ID")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# Initialize the client

client = openai.Client()

# Aggiungere constanti key GPT

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
    # Il percorso dove il file verrà salvato sul server
    image_path = os.path.join(temp_dir, file.filename)

    # Scrivi il file caricato nel percorso specificato
    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        print(f"Errore durante la scrittura del file: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nella scrittura del file: {e}")

    landmarks = detect_landmarks(image_path)

    if len(landmarks) > 0:
        landmarks_df = landmarks_combination_df(landmarks)
        prediction = au_pred_model.predict(landmarks_df)
        emotion = predict_emotion(prediction[0])

        aus_df = handle_au_and_emotions(prediction[0], emotion)

        process_and_analyze_aus(aus_df, arousal_valence_pred_model)

    # Restituisci informazioni sul file ricevuto
    return {"esito": "ok"}

@app.post("/upload-audio")
async def upload_audio(auth_id: str = Form(...), file: UploadFile = File(...)):
    if auth_id != ACCEPTED_AUTH_ID:
        raise HTTPException(status_code=401, detail="Unauthorized")

    permanent_audio_path = "src/audio"  # Definisci un percorso permanente dove salvare i file
    if not os.path.exists(permanent_audio_path):
        os.makedirs(permanent_audio_path)  # Crea la directory se non esiste

    file_path = os.path.join(permanent_audio_path, file.filename)
    try:
        # Salva il file caricato
        with open(file_path, "wb") as f:
            contents = await file.read()  # Assicurati di leggere il contenuto del file in modo asincrono
            f.write(contents)

        print(f"File ricevuto e salvato: {file.filename}")
        print(f"Percorso del file salvato: {file_path}")

        # Carica e converte il file audio
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        converted_file_path = os.path.splitext(file_path)[0] + "_converted.wav"
        audio.export(converted_file_path, format="wav")

        # Processo di conversione da audio a testo utilizzando il file WAV convertito
        recognizer = KaldiRecognizer(model, 16000)
        with open(converted_file_path, 'rb') as f:
            while True:
                data = f.read(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    pass

        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        print("Testo riconosciuto:", text)

        response = await handle_thread_creation_and_processing(text, ASSISTANT_ID)
        print("Risposta:", response)
        return {"esito": "ok", "testo": text, "risposta": response}
    except Exception as e:
        print(f"Errore durante il salvataggio o l'elaborazione del file: {e}")
        return JSONResponse(status_code=500, content={"esito": "errore", "dettaglio": str(e)})

async def handle_thread_creation_and_processing(text, my_assistant_id):
    # Crea un thread con il messaggio
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ]
    )

    # Crea una run per elaborare il thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=my_assistant_id
    )

    # Aspetta il completamento della run
    while run.status != 'completed':
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.status == 'closed':
            return {"status": run.status, "messages": None}
        time.sleep(1)

    # Recupera i messaggi dal thread
    thread_messages = client.beta.threads.messages.list(thread.id)
    response = thread_messages.data[0].content[0].text.value

    # Puoi scegliere di ritornare l'intera lista dei messaggi o elaborarla ulteriormente
    return {"status": run.status, "output": response, "input": text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
