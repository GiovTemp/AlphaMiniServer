# Usa un'immagine Python ufficiale come immagine di base
FROM python:3.9

# Imposta una directory di lavoro all'interno del container
WORKDIR /app

# Copia i file dei requisiti e installa le dipendenze
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dei file del progetto nella directory di lavoro del container
COPY . .

# Espone la porta su cui l'app sarà disponibile
EXPOSE 8000

# Comando per eseguire l'applicazione utilizzando uvicorn
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
