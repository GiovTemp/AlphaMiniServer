import numpy as np
import sys
sys.path.append('src')
from utils import load_json_data, save_data_to_json
from datetime import datetime

def analyze_frames(aus_array, model):
    # Crea una sequenza ripetendo il frame
    aus_np = np.array(aus_array)
    au_sequence = aus_np.reshape(1, 10, -1)

    predictions = model.predict(au_sequence)

    # Estrai arousal e valence dalle previsioni
    arousal, valence = predictions[0][0], predictions[0][1]

    return arousal, valence


def process_and_analyze_aus(arousal_valence_pred_model):
    # Load AU data from file
    aus_array = load_json_data("data/aus.json")

    # If there are more than 10 AUs, remove the oldest one (FIFO)
    if len(aus_array) > 10:
        aus_array.pop(0)

    # If we have collected 10 frames, analyze them
    if len(aus_array) == 10:
        av_json_file_path = "data/arousal_valence_data.json"
        # Load existing arousal and valence data
        av_data = load_json_data(av_json_file_path)
        arousal, valence = analyze_frames(aus_array, arousal_valence_pred_model)
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
