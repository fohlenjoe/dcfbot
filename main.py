from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

app = FastAPI()

# OpenAI API-Key über Environment Variable
openai.api_key = os.getenv("OPENAI_API_KEY")

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

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        max_tokens=500,
        temperature=0.7
    )

    swot = response.choices[0].message.content
    return { "swot": swot }
