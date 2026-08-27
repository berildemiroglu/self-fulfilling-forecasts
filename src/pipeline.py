import numpy as np

def generate_fundamental_process(F0, mu, sigma, n_rounds, seed):
    """
    Generate the fundamental value process F_t as a discretized
    geometric random walk: F_t = F_{t-1} * exp(mu + sigma * z_t).

    Parameters:
        F0 (float): starting fundamental value
        mu (float): per-round drift
        sigma (float): per-round volatility
        n_rounds (int): number of rounds to generate
        seed (int): random seed, for reproducibility across epsilon values

    Returns:
        np.array of length n_rounds+1 (includes F0 at index 0)
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_rounds)  # draw all shocks at once, same seed -> same path

    F = np.zeros(n_rounds + 1)
    F[0] = F0
    for t in range(1, n_rounds + 1):
        F[t] = F[t - 1] * np.exp(mu + sigma * z[t - 1])

    return F


def compute_price(F_t, agent_forecasts, epsilon):
    """
    Compute the realized market price as a convex combination
    of the fundamental value and the aggregated agent forecast.

    Parameters:
        F_t (float): fundamental value at time t
        agent_forecasts (list of float): each agent's forecast from t-1
        epsilon (float): responsiveness parameter, in [0, 1]

    Returns:
        float: realized price P_t
    """
    agg_forecast = np.mean(agent_forecasts)
    P_t = (1 - epsilon) * F_t + epsilon * agg_forecast
    return P_t


# Quick sanity check when running this file directly
if __name__ == "__main__":
    # Test 1: fundamental process generation
    F = generate_fundamental_process(F0=100, mu=0.001, sigma=0.02, n_rounds=40, seed=1)
    print("First 5 fundamental values:", F[:5])
    print("Last fundamental value:", F[-1])
    print()

    # Test 2: price computation, boundary check (epsilon=0 should equal F_t)
    fake_forecasts = [102, 98, 100, 105, 99]
    p_eps0 = compute_price(F_t=100, agent_forecasts=fake_forecasts, epsilon=0)
    p_eps1 = compute_price(F_t=100, agent_forecasts=fake_forecasts, epsilon=1)
    print(f"epsilon=0 -> P_t = {p_eps0} (should equal F_t = 100)")
    print(f"epsilon=1 -> P_t = {p_eps1} (should equal mean of forecasts = {np.mean(fake_forecasts)})")