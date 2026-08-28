import json
from dotenv import load_dotenv
from openai import OpenAI
import os
import numpy as np
from pipeline import generate_fundamental_process, compute_price

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PERSONA_2 = """You are a short-horizon mean-reverting market forecaster. You believe prices that have moved sharply away from their recent average — the last several rounds — tend to snap back toward it quickly. Focus specifically on short-term overshoots: look at the most recent handful of prices, judge how far the latest price has strayed from that short-run average, and forecast a partial correction toward it. Respond in JSON: {"reasoning": "...", "forecast": <number>}."""

PERSONA_3 = """You are a long-horizon, fundamentals-anchored market forecaster. You believe there is a stable underlying value that the market oscillates around over long periods, and that short-term swings — including multi-round trends — are noise around it. Estimate that long-run anchor using the entire price history available to you, not just recent rounds, and forecast a value close to that long-run anchor, adjusting only slightly for the current price. Deliberately ignore short-term momentum and short-run reversals. Respond in JSON: {"reasoning": "...", "forecast": <number>}."""


def get_forecast(persona_prompt, price_history):
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": f"Historical prices: {price_history}"}
        ]
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed["forecast"]


# Use a fixed, strongly-trending synthetic price series (not F_t directly)
# to test whether Persona 2 (short-run reversion) and Persona 3 (long-run
# anchoring) meaningfully diverge when there IS a clear trend to react to.
# Simulate 5 slightly different versions of a rising-trend scenario (proxy for seeds).
np.random.seed(42)
base_trend = np.array([100, 102, 104.5, 107, 110.5, 113, 116.5, 120])

diffs = []
for seed in range(1, 6):
    noise = np.random.normal(0, 0.5, size=len(base_trend))
    price_history = list(base_trend + noise)
    price_history = [round(p, 2) for p in price_history]

    forecast_2 = get_forecast(PERSONA_2, price_history)
    forecast_3 = get_forecast(PERSONA_3, price_history)
    diff = forecast_2 - forecast_3
    diffs.append(diff)

    print(f"Seed {seed}: prices={price_history}")
    print(f"  Persona 2 = {forecast_2:.2f}, Persona 3 = {forecast_3:.2f}, diff = {diff:.2f}\n")

print(f"Mean difference across 5 seeds: {np.mean(diffs):.2f}")
print(f"Std of difference: {np.std(diffs):.2f}")
print(f"All differences same sign (consistent direction)? {all(d > 0 for d in diffs) or all(d < 0 for d in diffs)}")