import asyncio
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def get_response(prompt_number):
    response = await client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "user", "content": f"Say hello, this is request number {prompt_number}"}
        ]
    )
    return response.choices[0].message.content


async def run_parallel_test(n_calls):
    tasks = [get_response(i) for i in range(n_calls)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    n = 20  # same order of magnitude as one round's worth of calls (5 agents x 4 personas-ish)

    start = time.time()
    results = asyncio.run(run_parallel_test(n))
    elapsed = time.time() - start

    print(f"Completed {n} calls in {elapsed:.2f} seconds")
    print(f"Average: {elapsed / n:.2f} sec/call (parallel)")
    print(f"\nSample response: {results[0]}")