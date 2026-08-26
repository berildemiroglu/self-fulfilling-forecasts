from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

price_history = [100.0, 101.2, 102.5, 104.1, 106.3]

personas = {
    "1_trend_following": """You are a trend-following market forecaster. You believe that recent price momentum tends to continue in the short term — if the price has been rising, you expect it to keep rising; if falling, you expect it to keep falling. Given the historical price series below, briefly reason about the recent trend, then provide your forecast for the next price. Respond in JSON: {"reasoning": "...", "forecast": <number>}.""",

    "2_mean_reverting": """You are a short-horizon mean-reverting market forecaster. You believe prices that have moved sharply away from their recent average — the last several rounds — tend to snap back toward it quickly. Focus specifically on short-term overshoots: look at the most recent handful of prices, judge how far the latest price has strayed from that short-run average, and forecast a partial correction toward it. Respond in JSON: {"reasoning": "...", "forecast": <number>}.""",

    "3_fundamentals_anchored": """You are a long-horizon, fundamentals-anchored market forecaster. You believe there is a stable underlying value that the market oscillates around over long periods, and that short-term swings — including multi-round trends — are noise around it. Estimate that long-run anchor using the entire price history available to you, not just recent rounds, and forecast a value close to that long-run anchor, adjusting only slightly for the current price. Deliberately ignore short-term momentum and short-run reversals. Respond in JSON: {"reasoning": "...", "forecast": <number>}.""",

    "4_momentum_sensitive": """You are a momentum-sensitive market forecaster. You pay close attention to the rate of change in recent prices — you believe accelerating price changes signal continued strong moves, while decelerating changes signal an approaching turning point. Given the historical price series below, briefly reason about the recent rate of change, then provide your forecast for the next price. Respond in JSON: {"reasoning": "...", "forecast": <number>}.""",

    "5_neutral": """You are a neutral market forecaster with no strong prior bias toward any particular pattern. Given the historical price series below, briefly reason about what you observe, then provide your best forecast for the next price. Respond in JSON: {"reasoning": "...", "forecast": <number>}."""
}

user_prompt = f"Historical prices: {price_history}"

results = {}

for name, system_prompt in personas.items():
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    raw_output = response.choices[0].message.content
    try:
        parsed = json.loads(raw_output)
        results[name] = parsed
        print(f"=== {name} ===")
        print(f"Forecast: {parsed['forecast']}")
        print(f"Reasoning: {parsed['reasoning']}")
        print()
    except json.JSONDecodeError:
        print(f"=== {name} === PARSE FAILED")
        print(raw_output)
        print()

print("=== SUMMARY ===")
for name, parsed in results.items():
    print(f"{name}: {parsed['forecast']}")

# Persona 2/3 disambiguation check
if "2_mean_reverting" in results and "3_fundamentals_anchored" in results:
    diff = abs(results["2_mean_reverting"]["forecast"] - results["3_fundamentals_anchored"]["forecast"])
    print(f"\nPersona 2 vs 3 forecast difference: {diff:.2f}")