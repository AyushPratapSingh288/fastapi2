import string
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import nltk
import re
import logging
import keras
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

nltk.download('stopwords')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tokenizer(filepath):
    try:
        with open(filepath, 'rb') as file:
            tokenizer = pickle.load(file)
        logger.info("Tokenizer loaded successfully!")
        return tokenizer
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
    except OSError as e:
        logger.error(f"Error loading tokenizer: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return None


def load_model(filepath):
    try:
        model = keras.models.load_model(filepath)
        logger.info("Model loaded successfully!")
        return model
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
    except OSError as e:
        logger.error(f"Error loading model: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return None


token_load = load_tokenizer('tokenizer.pickle')
model = load_model("model.h5")

stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))


class InputData(BaseModel):
    feature: str


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post("/predict")
def predict_banknote(text: InputData):
    logger.info(f"Received text: {text}")
    text_processed = text.feature.lower()
    text_processed = re.sub(r'https?://\S+|www\.\S+', '', text_processed)
    text_processed = re.sub(r'<.*?>+', '', text_processed)
    text_processed = re.sub(r'[%s]' % re.escape(string.punctuation), '', text_processed)
    text_processed = re.sub(r'\n', '', text_processed)
    text_processed = re.sub(r'\w*\d\w*', '', text_processed)
    text_processed = [word for word in text_processed.split(' ') if word not in stopword]
    text_processed = " ".join(text_processed)
    text_processed = [stemmer.stem(word) for word in text_processed.split(' ')]
    text_processed = " ".join(text_processed)

    seq = token_load.texts_to_sequences([text_processed])
    padded = pad_sequences(seq, maxlen=300)
    pred = model.predict(padded)

    logger.info(f"Prediction result: {pred}")
    prediction = pred.tolist()[0][0]
    if prediction < 0.5:
        result = "Normal"
    else:
        result = "Need expert attention"

    return {"prediction": result, "confidence": prediction}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
