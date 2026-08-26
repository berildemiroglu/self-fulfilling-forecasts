from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Örnek bir fiyat geçmişi (varsayımsal, test amaçlı)
price_history = [100.0, 101.2, 102.5, 104.1, 106.3]

system_prompt = """You are a trend-following market forecaster. You believe that recent price momentum tends to continue in the short term — if the price has been rising, you expect it to keep rising; if falling, you expect it to keep falling. Given the historical price series below, briefly reason about the recent trend, then provide your forecast for the next price. Respond in JSON: {"reasoning": "...", "forecast": <number>}."""

user_prompt = f"Historical prices: {price_history}"

response = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

raw_output = response.choices[0].message.content
print("RAW OUTPUT:")
print(raw_output)
print()

# JSON olarak ayrıştırmayı dene
try:
    parsed = json.loads(raw_output)
    print("PARSED SUCCESSFULLY:")
    print("Reasoning:", parsed["reasoning"])
    print("Forecast:", parsed["forecast"])
except json.JSONDecodeError:
    print("JSON PARSE FAILED — model may have added extra text (e.g. markdown code fences)")