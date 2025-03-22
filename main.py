from fastapi import FastAPI
from pydantic import BaseModel
import os
from huggingface_hub import InferenceClient

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HF_TOKEN
)

class SWOTRequest(BaseModel):
    ticker: str
    news: list
    roe: float
    dcf: float
    price: float

@app.post("/swot")
async def generate_swot(data: SWOTRequest):
    prompt = f"""
    Erstelle eine prägnante SWOT-Analyse für das börsennotierte Unternehmen mit dem Ticker {data.ticker}.

    Nutze folgende Finanzdaten:
    - ROE: {data.roe}%
    - Innerer Wert (DCF): {data.dcf} USD
    - Aktueller Kurs: {data.price} USD

    Berücksichtige diese Schlagzeilen der letzten 24 Stunden:
    - {data.news[0]}
    - {data.news[1]}
    - {data.news[2]}

    Gib die SWOT-Analyse auf Deutsch als einfache Tabelle mit 4 Punkten pro Kategorie (Stärken, Schwächen, Chancen, Risiken) zurück.
    """

    response = client.text_generation(prompt, max_new_tokens=400, temperature=0.7)
    return { "swot": response }
