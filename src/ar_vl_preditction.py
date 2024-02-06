import numpy as np
import sys

sys.path.append('src')
from utils import load_json_data, save_data_to_json
from datetime import datetime


def analyze_frames(au_values, model):

    predictions = model.predict(au_values)

    # Estrai arousal e valence dalle previsioni
    arousal, valence = predictions[0][0], predictions[0][1]

    return arousal, valence


def process_and_analyze_aus(au_values, arousal_valence_pred_model):
    # Load AU data from file
    av_json_file_path = "data/arousal_valence_data.json"
    # Load existing arousal and valence data
    av_data = load_json_data(av_json_file_path)
    arousal, valence = analyze_frames(au_values, arousal_valence_pred_model)
    current_time = datetime.now().isoformat()

    # Create a new record
    new_record = {
        "timestamp": current_time,
        "arousal": arousal,
        "valence": valence
    }

    # Append the new record to the existing data
    av_data.append(new_record)

    # Save the updated data back to the JSON file
    save_data_to_json(av_data, av_json_file_path)
