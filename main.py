from fastapi import FastAPI, Form
from pydantic import BaseModel
from typing import List, Literal

# My custom functions
from utils import text_cleaning, text_lemamtizing, text_vectorizing, predict_new




app = FastAPI()




# Dictionary for mapping the label to text
map_label = {
            0: 'Negative',
            1: 'Positive',
            2: 'Neutral'
        }


class DataInput(BaseModel):
    text: str
    method: Literal['BOW', 'TF-IDF', 'W2V', 'FT', 'GloVe'] = 'TF-IDF'


@app.post('/predict')
async def tweet_clf(data: DataInput):
    

    # Cleaning
    cleaned_text = text_cleaning(text=data.text)

    # Lemmatization
    cleaned_text = text_lemamtizing(text=cleaned_text)

    # Vectorizing
    X_processed = text_vectorizing(text=cleaned_text, method=data.method)

    # Model
    y_pred = predict_new(X_new=X_processed, method=data.method)

    # Map integer to Class Text
    final_pred = map_label.get(y_pred)

    return {f'Prediction is: {final_pred}'}

