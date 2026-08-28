import asyncio
import json
import time
import csv
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
from pipeline import generate_fundamental_process, compute_price

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PERSONAS = {
    "trend_following": """You are a trend-following market forecaster. You believe that recent price momentum tends to continue in the short term — if the price has been rising, you expect it to keep rising; if falling, you expect it to keep falling. Given the historical price series below, briefly reason about the recent trend, then provide your forecast for the next price. Respond in JSON: {"reasoning": "...", "forecast": <number>}.""",

    "mean_reverting": """You are a short-horizon mean-reverting market forecaster. You believe prices that have moved sharply away from their recent average — the last several rounds — tend to snap back toward it quickly. Focus specifically on short-term overshoots: look at the most recent handful of prices, judge how far the latest price has strayed from that short-run average, and forecast a partial correction toward it. Respond in JSON: {"reasoning": "...", "forecast": <number>}.""",

    "fundamentals_anchored": """You are a long-horizon, fundamentals-anchored market forecaster. You believe there is a stable underlying value that the market oscillates around over long periods, and that short-term swings — including multi-round trends — are noise around it. Estimate that long-run anchor using the entire price history available to you, not just recent rounds, and forecast a value close to that long-run anchor, adjusting only slightly for the current price. Deliberately ignore short-term momentum and short-run reversals. Respond in JSON: {"reasoning": "...", "forecast": <number>}.""",

    "momentum_sensitive": """You are a momentum-sensitive market forecaster. You pay close attention to the rate of change in recent prices — you believe accelerating price changes signal continued strong moves, while decelerating changes signal an approaching turning point. Given the historical price series below, briefly reason about the recent rate of change, then provide your forecast for the next price. Respond in JSON: {"reasoning": "...", "forecast": <number>}.""",

    "neutral": """You are a neutral market forecaster with no strong prior bias toward any particular pattern. Given the historical price series below, briefly reason about what you observe, then provide your best forecast for the next price. Respond in JSON: {"reasoning": "...", "forecast": <number>}."""
}

EPSILON_GRID = [0, 0.2, 0.4, 0.6, 0.8]
SEEDS = list(range(1, 9))
N_ROUNDS = 40


async def get_agent_forecast_async(persona_prompt, price_history, max_retries=2):
    if len(price_history) == 0:
        user_prompt = "No price history yet. This is the first round. Provide an initial forecast around 100."
    else:
        user_prompt = f"Historical prices: {price_history}"

    for attempt in range(max_retries + 1):
        response = await client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw_output = response.choices[0].message.content
        try:
            parsed = json.loads(raw_output)
            return parsed["forecast"]
        except (json.JSONDecodeError, KeyError):
            if attempt == max_retries:
                fallback = price_history[-1] if price_history else 100.0
                return fallback


async def get_all_forecasts_parallel(price_history):
    tasks = [get_agent_forecast_async(p, price_history) for p in PERSONAS.values()]
    return list(await asyncio.gather(*tasks))


async def run_single_condition(epsilon, seed):
    """Run one full 40-round market simulation for a given (epsilon, seed) pair."""
    F = generate_fundamental_process(F0=100, mu=0.001, sigma=0.02, n_rounds=N_ROUNDS, seed=seed)
    price_history = []

    forecasts_prev = await get_all_forecasts_parallel([])

    round_data = []
    for t in range(1, N_ROUNDS + 1):
        P_t = compute_price(F[t], forecasts_prev, epsilon)
        price_history.append(P_t)

        round_data.append({
            "epsilon": epsilon,
            "seed": seed,
            "round": t,
            "F_t": F[t],
            "P_t": P_t,
            "deviation": P_t - F[t],
            "forecasts": forecasts_prev.copy()
        })

        forecasts_prev = await get_all_forecasts_parallel(price_history)

    return round_data


async def run_full_experiment():
    all_data = []
    total_conditions = len(EPSILON_GRID) * len(SEEDS)
    completed = 0

    start = time.time()

    for epsilon in EPSILON_GRID:
        for seed in SEEDS:
            condition_start = time.time()
            round_data = await run_single_condition(epsilon, seed)
            all_data.extend(round_data)
            completed += 1
            condition_time = time.time() - condition_start

            print(f"[{completed}/{total_conditions}] epsilon={epsilon}, seed={seed} "
                  f"done in {condition_time:.1f}s")

    total_time = time.time() - start
    print(f"\nFull experiment completed in {total_time / 60:.1f} minutes")

    # Save to CSV
    output_path = "data/main_experiment_luna.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epsilon", "seed", "round", "F_t", "P_t", "deviation", "forecasts"])
        writer.writeheader()
        for row in all_data:
            row_copy = row.copy()
            row_copy["forecasts"] = json.dumps(row_copy["forecasts"])
            writer.writerow(row_copy)

    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_full_experiment())