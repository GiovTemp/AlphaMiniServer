import sys

sys.path.append('src')
from utils import save_data_to_json, load_json_data
from datetime import datetime
import pandas as pd


def limit_list_to_10_items(lst):
    return lst[-10:]


def handle_au_and_emotions(au_values, emotion):
    """
    Handle and store new AU values and emotion data.

    Loads existing AU and emotion data, appends new values while maintaining a maximum of 10 records (FIFO approach),
    and saves the updated data back to JSON files.

    :param au_values: New AU data to be added.
    :param emotion: New emotion data to be added, stored with a timestamp.
    """
    # Load or initialize existing arrays for AU and Emotions
    aus_array = load_json_data("data/aus.json")
    emotions_array = load_json_data("data/emotions.json")

    # Convert NumPy arrays to Python lists
    #au_values_list = au_values.tolist()  # Convert au_values to a Python list

    # Creazione di un DataFrame con le Action Units
    aus_df = pd.DataFrame([au_values])

    # Aggiunta di colonne per le emozioni
    emotions_columns = ['happiness', 'sadness', 'surprise', 'fear', 'anger', 'neutral']
    aus_df[emotions_columns] = 0
    aus_df[emotion] = 1

    # Add new values and apply FIFO
    aus_array.append(au_values)
    #aus_array = limit_list_to_10_items(aus_array)

    emotions_array.append({"timestamp": datetime.now().isoformat(), "emotion": emotion})
    #emotions_array = limit_list_to_10_items(emotions_array)

    # Save updated data to JSON files
    save_data_to_json(aus_array, "data/aus.json")
    save_data_to_json(emotions_array, "data/emotions.json")

    return aus_df


def predict_emotion(prediction):
    """
    Predicts the most likely emotion from the given AU prediction values.

    Uses a predefined set of AU combinations associated with different emotions. The emotion with the highest number of matching AU values (above a threshold of 0.48) is selected as the predicted emotion.

    :param prediction: A list of AU prediction values.
    :return: The predicted emotion as a string.
    """
    emotions = {'happiness': [4, 9], 'sadness': [0, 2, 11], 'surprise': [0, 1, 3, 17], 'fear': [0, 1, 2, 3, 5, 13, 17],
                'anger': [2, 3, 5, 14]}

    emotion = 'neutral'
    diff_temp = 0

    for key, value in emotions.items():
        sum = 0
        v_len = len(value) / 2

        for v in value:
            if prediction[v] >= 0.48:
                sum += 1

        diff = sum - v_len

        if diff >= diff_temp:
            emotion = key
            diff_temp = sum - v_len

    return emotion
