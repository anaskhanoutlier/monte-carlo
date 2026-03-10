

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────
# SECTION 1: PI ESTIMATION (Classic Monte Carlo)
# ─────────────────────────────────────────

def estimate_pi(n_samples=100000):
    """
    Estimate π using Monte Carlo:
    - Throw random darts at unit square
    - Count those inside unit circle (x²+y²≤1)
    - π ≈ 4 × (inside_circle / total)
    """
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside = x**2 + y**2 <= 1

    cumulative_pi = 4 * np.cumsum(inside) / np.arange(1, n_samples + 1)
    final_pi = cumulative_pi[-1]
    error = abs(final_pi - np.pi)

    return x, y, inside, cumulative_pi, final_pi, error


# ─────────────────────────────────────────
# SECTION 2: INTEGRATION BY MONTE CARLO
# ─────────────────────────────────────────

def monte_carlo_integration(f, a, b, n_samples=100000):
    """
    Monte Carlo Integration:
    ∫f(x)dx ≈ (b-a) × E[f(X)]  where X ~ Uniform(a,b)
    
    Much more efficient for high-dimensional integrals
    compared to deterministic methods.
    """
    x = np.random.uniform(a, b, n_samples)
    f_vals = f(x)
    mc_estimate = (b - a) * np.mean(f_vals)
    mc_std      = (b - a) * np.std(f_vals) / np.sqrt(n_samples)

    return mc_estimate, mc_std, x, f_vals


def high_dim_volume_estimation(dims=5, n_samples=500000):
    """
    Volume of a unit hypersphere in d dimensions.
    Exact formula: V_d = π^(d/2) / Γ(d/2 + 1)
    Monte Carlo: sample uniform in [-1,1]^d, count points with ||x|| ≤ 1
    """
    from scipy.special import gamma
    results = []
    for d in range(2, dims + 1):
        X = np.random.uniform(-1, 1, (n_samples, d))
        inside = np.sum(X**2, axis=1) <= 1
        mc_vol = (2**d) * np.mean(inside)
        exact = (np.pi**(d/2)) / gamma(d/2 + 1)
        results.append({
            'dims': d,
            'mc_volume': mc_vol,
            'exact_volume': exact,
            'error': abs(mc_vol - exact),
            'pct_error': abs(mc_vol - exact) / exact * 100
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────
# SECTION 3: LAW OF LARGE NUMBERS
# ─────────────────────────────────────────

def law_of_large_numbers():
    """
    Demonstrate LLN:
    Sample mean → Population mean as n → ∞
    Uses: Coin flip, Dice roll, Exponential random variable
    """
    n_max = 10000
    ns    = np.arange(1, n_max + 1)

    results = {}

    # Coin flip (Bernoulli p=0.5, E[X]=0.5)
    coin = np.random.randint(0, 2, n_max)
    results['Coin Flip (Expected=0.5)'] = {
        'data': np.cumsum(coin) / ns,
        'expected': 0.5
    }

    # Dice roll (E[X] = 3.5)
    dice = np.random.randint(1, 7, n_max).astype(float)
    results['Dice Roll (Expected=3.5)'] = {
        'data': np.cumsum(dice) / ns,
        'expected': 3.5
    }

    # Exponential (λ=2, E[X] = 1/λ = 0.5)
    expo = np.random.exponential(scale=0.5, size=n_max)
    results['Exponential λ=2 (Expected=0.5)'] = {
        'data': np.cumsum(expo) / ns,
        'expected': 0.5
    }

    # Normal (μ=3, σ=5)
    norm = np.random.normal(3, 5, n_max)
    results['Normal μ=3, σ=5 (Expected=3)'] = {
        'data': np.cumsum(norm) / ns,
        'expected': 3.0
    }

    return ns, results


# ─────────────────────────────────────────
# SECTION 4: RANDOM WALK
# ─────────────────────────────────────────

def random_walk_1d(n_steps=1000, n_walks=5):
    """1D Random Walk: each step ±1 with equal probability."""
    walks = []
    for _ in range(n_walks):
        steps = np.random.choice([-1, 1], size=n_steps)
        walk  = np.concatenate([[0], np.cumsum(steps)])
        walks.append(walk)
    return np.array(walks)


def random_walk_2d(n_steps=2000, n_walks=5):
    """2D Random Walk (North/South/East/West)."""
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    walks_2d = []
    for _ in range(n_walks):
        x_pos, y_pos = [0], [0]
        x, y = 0, 0
        for _ in range(n_steps):
            dx, dy = directions[np.random.randint(4)]
            x += dx
            y += dy
            x_pos.append(x)
            y_pos.append(y)
        walks_2d.append((np.array(x_pos), np.array(y_pos)))
    return walks_2d


# ─────────────────────────────────────────
# SECTION 5: MARKOV CHAIN
# ─────────────────────────────────────────

def markov_chain_weather():
    """
    Markov Chain — Weather Model
    States: Sunny (0), Cloudy (1), Rainy (2)
    Transition Matrix P where P[i,j] = P(j|i)
    Stationary Distribution: π such that πP = π
    """
    # Transition matrix
    P = np.array([
        [0.6, 0.3, 0.1],   # Sunny → Sunny/Cloudy/Rainy
        [0.3, 0.4, 0.3],   # Cloudy → Sunny/Cloudy/Rainy
        [0.2, 0.3, 0.5],   # Rainy → Sunny/Cloudy/Rainy
    ])

    state_names = ['Sunny', 'Cloudy', 'Rainy']

    # Simulate chain
    n_steps = 365  # simulate 1 year
    states = np.zeros(n_steps, dtype=int)
    states[0] = 0  # start sunny

    for t in range(1, n_steps):
        states[t] = np.random.choice(3, p=P[states[t-1]])

    # Compute empirical stationary distribution
    empirical = np.array([np.mean(states == i) for i in range(3)])

    # Theoretical stationary distribution (eigenvector method)
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1))
    stationary = np.abs(eigenvectors[:, stationary_idx])
    stationary = stationary / stationary.sum()

    print("\n━━━ MARKOV CHAIN — WEATHER MODEL ━━━")
    print("  Transition Matrix P:")
    print(f"  {'':>8}", end='')
    for s in state_names: print(f"{s:>10}", end='')
    print()
    for i, row in enumerate(P):
        print(f"  {state_names[i]:<8}", end='')
        for val in row: print(f"{val:>10.2f}", end='')
        print()

    print(f"\n  Theoretical Stationary π: ", end='')
    for name, p in zip(state_names, stationary):
        print(f"{name}={p:.4f} ", end='')
    print()
    print(f"  Empirical (365 steps):    ", end='')
    for name, p in zip(state_names, empirical):
        print(f"{name}={p:.4f} ", end='')
    print()

    return P, states, state_names, stationary, empirical


# ─────────────────────────────────────────
# SECTION 6: BAYESIAN ESTIMATION (Monte Carlo)
# ─────────────────────────────────────────

def bayesian_coin_estimation(n_flips=100, true_p=0.65):
    """
    Bayesian estimation of coin bias using Monte Carlo.
    Prior: Beta(2, 2) (weakly informative, centered at 0.5)
    Likelihood: Binomial
    Posterior: Beta(α + heads, β + tails) — conjugate
    """
    # Simulate flips
    flips = np.random.binomial(1, true_p, n_flips)
    heads = flips.sum()
    tails = n_flips - heads

    # Prior: Beta(2, 2)
    alpha_prior, beta_prior = 2, 2

    # Posterior: Beta(α + heads, β + tails)
    alpha_post = alpha_prior + heads
    beta_post  = beta_prior  + tails

    # MLE estimate
    mle = heads / n_flips

    # Posterior mean
    post_mean = alpha_post / (alpha_post + beta_post)

    # Posterior 95% credible interval
    cred_int = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

    print(f"\n━━━ BAYESIAN ESTIMATION — COIN BIAS ━━━")
    print(f"  True bias           : {true_p}")
    print(f"  Flips               : {n_flips}  (Heads={heads}, Tails={tails})")
    print(f"  MLE estimate        : {mle:.4f}")
    print(f"  Prior: Beta(2,2)    : Mean = {alpha_prior/(alpha_prior+beta_prior):.4f}")
    print(f"  Posterior Mean      : {post_mean:.4f}")
    print(f"  95% Credible Interval: [{cred_int[0]:.4f}, {cred_int[1]:.4f}]")
    print(f"  Posterior: Beta({alpha_post}, {beta_post})")

    return alpha_prior, beta_prior, alpha_post, beta_post, mle, post_mean, cred_int


# ─────────────────────────────────────────
# SECTION 7: FINANCIAL SIMULATION (Monte Carlo)
# ─────────────────────────────────────────

def stock_price_simulation(S0=100, mu=0.12, sigma=0.25,
                            T=1.0, dt=1/252, n_sims=1000):
    """
    Geometric Brownian Motion (GBM) for stock price:
    dS = μS dt + σS dW
    S(t+dt) = S(t) × exp((μ - σ²/2)dt + σ√dt Z)  where Z ~ N(0,1)
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)

    # Simulate all paths at once (vectorized)
    Z = np.random.standard_normal((n_steps - 1, n_sims))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    prices = np.zeros((n_steps, n_sims))
    prices[0] = S0
    prices[1:] = S0 * np.exp(np.cumsum(log_returns, axis=0))

    final_prices = prices[-1]
    mean_final   = final_prices.mean()
    std_final    = final_prices.std()
    var_95       = np.percentile(final_prices, 5)   # 5th percentile = VaR 95%
    prob_profit  = np.mean(final_prices > S0) * 100

    print(f"\n━━━ STOCK PRICE SIMULATION (GBM) ━━━")
    print(f"  Initial Price S₀    : ₹{S0}")
    print(f"  Annual Drift μ      : {mu*100:.1f}%")
    print(f"  Annual Volatility σ : {sigma*100:.1f}%")
    print(f"  Horizon             : {T} year ({n_steps} days)")
    print(f"  Simulations         : {n_sims}")
    print(f"  Expected Final Price: ₹{mean_final:.2f} ± ₹{std_final:.2f}")
    print(f"  VaR (95%)           : ₹{var_95:.2f}")
    print(f"  Probability of Profit: {prob_profit:.1f}%")

    return t, prices, final_prices


# ─────────────────────────────────────────
# SECTION 8: VISUALIZATION
# ─────────────────────────────────────────

def visualize_all(x_pi, y_pi, inside, cumulative_pi,
                  ns, lln_results,
                  walks_1d, walks_2d,
                  P, states, state_names, stationary,
                  alpha_prior, beta_prior, alpha_post, beta_post, mle,
                  t, prices):

    sns.set_style("whitegrid")

    # ── Figure 1: Core Monte Carlo ──
    fig1 = plt.figure(figsize=(18, 12))
    fig1.suptitle("Monte Carlo Simulations & Computational Probability\n"
                  "BSc Mathematics + Statistics | Python · NumPy · SciPy · Matplotlib",
                  fontsize=13, fontweight='bold')
    gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.38)

    # π estimation
    ax = fig1.add_subplot(gs1[0, 0])
    n_show = 3000
    ax.scatter(x_pi[inside][:n_show], y_pi[inside][:n_show],
               color='steelblue', s=2, alpha=0.5, label='Inside')
    ax.scatter(x_pi[~inside][:1000], y_pi[~inside][:1000],
               color='tomato', s=2, alpha=0.5, label='Outside')
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, zorder=5)
    ax.set_aspect('equal')
    ax.set_title(f"Monte Carlo π Estimation\nπ ≈ {cumulative_pi[-1]:.5f}", fontweight='bold')
    ax.legend(fontsize=7, markerscale=4)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)

    # π convergence
    ax = fig1.add_subplot(gs1[0, 1])
    ax.semilogx(np.arange(1, len(cumulative_pi)+1), cumulative_pi,
                color='steelblue', linewidth=1.5, label='MC Estimate')
    ax.axhline(np.pi, color='red', linestyle='--', linewidth=2, label=f'True π = {np.pi:.5f}')
    ax.fill_between(np.arange(1, len(cumulative_pi)+1),
                    cumulative_pi - 0.1, cumulative_pi + 0.1, alpha=0.1, color='steelblue')
    ax.set_xlabel("Number of Samples (log scale)")
    ax.set_ylabel("π Estimate")
    ax.set_title("π Estimation Convergence\n(Law of Large Numbers)", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.5, 3.8)

    # LLN
    ax = fig1.add_subplot(gs1[0, 2])
    colors_lln = ['steelblue', 'green', 'tomato', 'purple']
    for (name, res), color in zip(lln_results.items(), colors_lln):
        ax.semilogx(ns, res['data'], color=color, linewidth=1.5, alpha=0.8, label=name[:20])
        ax.axhline(res['expected'], color=color, linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel("Sample Size n (log scale)")
    ax.set_ylabel("Running Mean")
    ax.set_title("Law of Large Numbers\n(Convergence to E[X])", fontweight='bold')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 1D Random Walk
    ax = fig1.add_subplot(gs1[1, 0])
    colors_walk = plt.cm.tab10(np.linspace(0, 1, len(walks_1d)))
    for i, walk in enumerate(walks_1d):
        ax.plot(walk, color=colors_walk[i], linewidth=1, alpha=0.8)
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Position")
    ax.set_title("1D Random Walk\n(5 Simulations)", fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2D Random Walk
    ax = fig1.add_subplot(gs1[1, 1])
    colors_2d = plt.cm.Set2(np.linspace(0, 1, len(walks_2d)))
    for (xp, yp), color in zip(walks_2d, colors_2d):
        ax.plot(xp, yp, color=color, linewidth=0.8, alpha=0.7)
        ax.scatter([xp[0]], [yp[0]], color=color, s=50, marker='o', zorder=5)
        ax.scatter([xp[-1]], [yp[-1]], color=color, s=80, marker='*', zorder=5)
    ax.scatter([0], [0], color='black', s=100, zorder=6, label='Start (★=End)')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("2D Random Walk\n(5 Simulations)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Markov Chain
    ax = fig1.add_subplot(gs1[1, 2])
    colors_mc = {'Sunny': '#f39c12', 'Cloudy': '#95a5a6', 'Rainy': '#3498db'}
    window = 30
    for i, (state_name, color) in enumerate(colors_mc.items()):
        is_state = (states == i).astype(float)
        smooth = np.convolve(is_state, np.ones(window)/window, mode='valid')
        ax.plot(smooth, color=color, linewidth=1.8, label=f'{state_name} (π={stationary[i]:.3f})')
        ax.axhline(stationary[i], color=color, linestyle='--', linewidth=1, alpha=0.4)
    ax.set_xlabel("Day")
    ax.set_ylabel("30-day Moving Avg Probability")
    ax.set_title("Markov Chain — Weather Model\n(Convergence to Stationary π)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.savefig("project15_monte_carlo.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ── Figure 2: Bayesian + Stock Simulation ──
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Bayesian Estimation & GBM Stock Simulation — Monte Carlo\n"
                  "Statistics | Python · NumPy · SciPy",
                  fontsize=12, fontweight='bold')

    # Bayesian posterior
    ax = axes[0]
    p_range = np.linspace(0, 1, 500)
    prior = stats.beta.pdf(p_range, alpha_prior, beta_prior)
    posterior = stats.beta.pdf(p_range, alpha_post, beta_post)
    ax.plot(p_range, prior, 'b-', linewidth=2, label=f'Prior Beta({alpha_prior},{beta_prior})')
    ax.fill_between(p_range, prior, alpha=0.15, color='blue')
    ax.plot(p_range, posterior, 'r-', linewidth=2.5, label=f'Posterior Beta({alpha_post},{beta_post})')
    ax.fill_between(p_range, posterior, alpha=0.15, color='red')
    ax.axvline(mle, color='green', linestyle='--', linewidth=2, label=f'MLE={mle:.3f}')
    ax.axvline(0.65, color='black', linestyle=':', linewidth=2, label='True p=0.65')
    ax.set_xlabel("Coin Bias p")
    ax.set_ylabel("Density")
    ax.set_title("Bayesian Estimation\nPrior → Posterior", fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Stock price paths
    ax = axes[1]
    n_show_paths = min(50, prices.shape[1])
    for i in range(n_show_paths):
        ax.plot(t, prices[:, i], alpha=0.15, linewidth=0.8,
                color='steelblue' if prices[-1, i] > prices[0, i] else 'tomato')
    ax.plot(t, prices.mean(axis=1), 'k-', linewidth=2.5, label='Mean Path')
    ax.plot(t, np.percentile(prices, 5, axis=1), 'r--', linewidth=2, label='5th Pctl (VaR)')
    ax.plot(t, np.percentile(prices, 95, axis=1), 'g--', linewidth=2, label='95th Pctl')
    ax.axhline(prices[0, 0], color='black', linestyle=':', linewidth=1)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price (₹)")
    ax.set_title(f"GBM Stock Simulation\n{prices.shape[1]} Paths, μ=12%, σ=25%", fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Final price distribution
    ax = axes[2]
    final = prices[-1, :]
    ax.hist(final, bins=60, density=True, color='steelblue', edgecolor='white',
            linewidth=0.5, alpha=0.75, label='Simulated')
    x_lognorm = np.linspace(final.min(), final.max(), 300)
    mu_log = np.log(prices[0,0]) + (0.12 - 0.5*0.25**2) * 1.0
    sigma_log = 0.25 * np.sqrt(1.0)
    ax.plot(x_lognorm, stats.lognorm.pdf(x_lognorm, sigma_log, scale=np.exp(mu_log)),
            'r-', linewidth=2.5, label='Log-Normal (Theoretical)')
    ax.axvline(final.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Mean=₹{final.mean():.1f}')
    ax.axvline(np.percentile(final, 5), color='orange', linestyle='--', linewidth=2,
               label=f'VaR(95%)=₹{np.percentile(final,5):.1f}')
    ax.set_xlabel("Final Stock Price (₹)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Final Prices\n(Log-Normal)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("project15_bayesian_gbm.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Plots: project15_monte_carlo.png, project15_bayesian_gbm.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 65)
    print("PROJECT 15: Monte Carlo Simulations & Computational Probability")
    print("BSc Mathematics + Statistics | IGNOU + AMU")
    print("Python · NumPy · SciPy · Matplotlib · Pandas")
    print("=" * 65)

    # π estimation
    print("\n━━━ π ESTIMATION ━━━")
    for n in [100, 1000, 10000, 100000]:
        x, y, inside, cum_pi, est, err = estimate_pi(n)
        print(f"  n={n:>7}:  π ≈ {est:.6f}  (error = {err:.2e})")

    x_pi, y_pi, inside, cum_pi, _, _ = estimate_pi(100000)

    # Monte Carlo integration
    print("\n━━━ MONTE CARLO INTEGRATION ━━━")
    integrals = [
        (lambda x: np.sin(x), 0, np.pi, 2.0,             "∫sin(x)dx [0,π]"),
        (lambda x: np.exp(-x**2), -3, 3, np.sqrt(np.pi), "∫exp(-x²)dx [-3,3]"),
        (lambda x: x**4 - x**2 + 1, 0, 2, 5.0667,        "∫(x⁴-x²+1)dx [0,2]"),
    ]
    for f, a, b, true_val, name in integrals:
        est, std, _, _ = monte_carlo_integration(f, a, b, 100000)
        print(f"  {name:<30}: {est:.6f} ± {std:.6f}  (exact={true_val:.4f})")

    # Hypersphere volumes
    print("\n━━━ HYPERSPHERE VOLUME ESTIMATION ━━━")
    df_vol = high_dim_volume_estimation(dims=5, n_samples=300000)
    print(df_vol[['dims','mc_volume','exact_volume','pct_error']].round(4).to_string(index=False))

    # LLN
    ns, lln_results = law_of_large_numbers()

    # Random walks
    walks_1d = random_walk_1d(n_steps=2000, n_walks=5)
    walks_2d = random_walk_2d(n_steps=2000, n_walks=5)

    # Markov chain
    P, states, state_names, stationary, empirical = markov_chain_weather()

    # Bayesian estimation
    alpha_prior, beta_prior, alpha_post, beta_post, mle, post_mean, cred_int = \
        bayesian_coin_estimation(n_flips=100, true_p=0.65)

    # Stock simulation
    t, prices, final_prices = stock_price_simulation(n_sims=500)

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_all(x_pi, y_pi, inside, cum_pi,
                  ns, lln_results,
                  walks_1d, walks_2d,
                  P, states, state_names, stationary,
                  alpha_prior, beta_prior, alpha_post, beta_post, mle,
                  t, prices)


if __name__ == "__main__":
    main()
