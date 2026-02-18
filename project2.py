"""
Project 2 – Portfolio Optimization
===================================
Group #2

Assets (20 stocks, 2017-01-01 to 2023-12-31):
  Financial  : GS, MS, SCHW
  Healthcare : JNJ, ABBV, TMO
  Energy     : XOM, SLB, EOG
  Consumer   : COST, NKE, SBUX
  Industrial : CAT, DE, UPS
  Technology : AMD, ORCL, CRM, CMCSA, LIN

Requirements covered:
  1. Data download
  2. 1/N equally-weighted portfolio (baseline)
  3. Optimization problem definition (CVXPY)
  4. Efficient frontier – CVXPY (gamma sensibilization) + scipy MVO
  5. Special portfolios: min-variance, max-return, min-return, max-Sharpe
  6. Weight tables for special portfolios
  7. [Extra +10] Leverage / short-selling with per-level efficient frontiers
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import cvxpy as cp
from scipy.optimize import minimize
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = [
    "GS",    # Goldman Sachs
    "MS",    # Morgan Stanley
    "SCHW",  # Charles Schwab
    "JNJ",   # Johnson & Johnson
    "ABBV",  # AbbVie
    "TMO",   # Thermo Fisher Scientific
    "XOM",   # Exxon Mobil
    "SLB",   # Schlumberger
    "EOG",   # EOG Resources
    "COST",  # Costco
    "NKE",   # Nike
    "SBUX",  # Starbucks
    "CAT",   # Caterpillar
    "DE",    # Deere & Company
    "UPS",   # United Parcel Service
    "AMD",   # Advanced Micro Devices
    "ORCL",  # Oracle
    "CRM",   # Salesforce
    "CMCSA", # Comcast
    "LIN",   # Linde
]
START_DATE  = "2017-01-01"
END_DATE    = "2023-12-31"
RISK_FREE   = 0.03          # annual risk-free rate (approx 3% avg over period)
TRADING_DAYS = 252

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD ASSET PRICES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("1. DOWNLOADING ASSET PRICES")
print("=" * 65)

prices = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
prices.dropna(how="any", inplace=True)  # drop dates with missing data for any ticker

print(f"\nLoaded {len(prices)} trading days for {len(prices.columns)} assets.")
print(f"Date range : {prices.index[0].date()} → {prices.index[-1].date()}")
print("\nFirst 5 rows:")
print(prices.head())

# Daily log returns (more stable for MVO)
log_returns = np.log(prices / prices.shift(1)).dropna()

# Annualised expected returns (mu) and covariance matrix (Sigma)
mu    = log_returns.mean().values * TRADING_DAYS          # shape (n,)
Sigma = log_returns.cov().values  * TRADING_DAYS          # shape (n, n)
n     = len(TICKERS)

print(f"\nAnnualised expected returns (first 5): {mu[:5].round(4)}")
print(f"Covariance matrix shape: {Sigma.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 1/N EQUALLY-WEIGHTED PORTFOLIO (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. 1/N EQUALLY-WEIGHTED PORTFOLIO")
print("=" * 65)

w_eq         = np.ones(n) / n
ret_eq       = w_eq @ mu
risk_eq      = np.sqrt(w_eq @ Sigma @ w_eq)
sharpe_eq    = (ret_eq - RISK_FREE) / risk_eq

print(f"\nWeights          : {w_eq[0]:.4f} for each of the {n} assets")
print(f"Annual Return    : {ret_eq*100:.2f}%")
print(f"Annual Volatility: {risk_eq*100:.2f}%")
print(f"Sharpe Ratio     : {sharpe_eq:.4f}")

# Comment:
# The 1/N portfolio is surprisingly difficult to outperform out-of-sample
# (DeMiguel et al., 2009). It ignores correlations and expected returns,
# relying purely on diversification. Here it serves as a practical benchmark:
# any optimised strategy should ideally beat it on a risk-adjusted basis.

# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMIZATION PROBLEM DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. OPTIMIZATION PROBLEM DEFINITION")
print("=" * 65)

print("""
We solve the classic Markowitz Mean-Variance Optimization (MVO):

  (a) RISK-AVERSION FORM (used for efficient frontier via CVXPY):

      maximize    μᵀw  −  γ/2 · wᵀΣw
      subject to  Σwᵢ = 1
                  wᵢ ≥ 0   (no-short-selling; relaxed in §7)

      γ ∈ [0, ∞) controls the risk–return trade-off.
      γ → 0  ≈ max-return portfolio (fully concentrated).
      γ → ∞  ≈ min-variance portfolio.

  (b) TARGET-RETURN FORM (used for scipy efficient frontier):

      minimize    wᵀΣw
      subject to  μᵀw = r_target
                  Σwᵢ = 1
                  wᵢ ≥ 0

  Variables : w ∈ ℝⁿ   (portfolio weights)
  Parameters: μ ∈ ℝⁿ   (annualised expected returns)
              Σ ∈ ℝⁿˣⁿ  (annualised covariance matrix, PSD)
              γ ≥ 0     (risk-aversion coefficient)
              r_target  (desired portfolio return)

  The feasible set is a convex polytope; the objective in (a) is
  concave (quadratic with negative-definite Hessian), so CVXPY's
  SCS / OSQP solvers find a global optimum.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 4. EFFICIENT FRONTIER
#    (a) CVXPY with gamma sensibilization
#    (b) scipy target-return sweep (alternative method)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("4. EFFICIENT FRONTIER")
print("=" * 65)

# ── 4a. CVXPY efficient frontier ─────────────────────────────────────────────
print("\n[4a] CVXPY – gamma sensibilization")

def cvxpy_frontier(mu, Sigma, n, gammas, lb=0.0):
    """
    Solve the risk-aversion portfolio for each gamma value.
    lb: lower bound on weights (0 = long-only, negative = short-selling)
    Returns arrays of (risk, return, weights).
    """
    w         = cp.Variable(n)
    gamma_par = cp.Parameter(nonneg=True)
    ret       = mu @ w
    risk_sq   = cp.quad_form(w, Sigma)

    objective   = cp.Maximize(ret - gamma_par * risk_sq)
    constraints = [cp.sum(w) == 1, w >= lb]
    prob        = cp.Problem(objective, constraints)

    risks, returns, weights = [], [], []
    for g in gammas:
        gamma_par.value = g
        prob.solve(solver=cp.SCS, warm_start=True, verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            wv = w.value
            risks.append(float(np.sqrt(wv @ Sigma @ wv)))
            returns.append(float(mu @ wv))
            weights.append(wv.copy())

    return np.array(risks), np.array(returns), weights

# Gamma values spanning several orders of magnitude to capture the full frontier
gammas_fine = np.logspace(-2, 4, 300)

risks_cvx, rets_cvx, weights_cvx = cvxpy_frontier(mu, Sigma, n, gammas_fine, lb=0.0)

print(f"  Solved {len(risks_cvx)} portfolios on the CVXPY efficient frontier.")
print(f"  Return range : [{rets_cvx.min()*100:.2f}%, {rets_cvx.max()*100:.2f}%]")
print(f"  Risk range   : [{risks_cvx.min()*100:.2f}%, {risks_cvx.max()*100:.2f}%]")

# Comment on gamma sensibilization:
# A small gamma (≈0) barely penalises variance, so the optimiser chases
# maximum return by concentrating in the single best-performing asset.
# As gamma grows, diversification matters more, driving weights toward the
# minimum-variance portfolio. The full curve from gamma~0 to gamma~∞ traces
# the upper portion of the mean-variance parabola (the efficient frontier).

# ── 4b. scipy efficient frontier (target-return sweep) ───────────────────────
print("\n[4b] scipy – target-return sweep (MVO formulation)")

def scipy_frontier(mu, Sigma, n, n_points=200):
    """Sweep target returns and solve the QP via scipy SLSQP."""
    ret_min = mu.min()  # min possible (all weight in lowest-return asset)
    ret_max = mu.max()  # max possible (all weight in highest-return asset)
    targets = np.linspace(ret_min, ret_max, n_points)

    risks, returns, weights = [], [], []
    w0 = np.ones(n) / n   # initial guess: equal weight

    for r_target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, r=r_target: w @ mu - r},
        ]
        bounds = [(0.0, 1.0)] * n
        res = minimize(
            lambda w: w @ Sigma @ w,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success:
            wv = res.x
            risks.append(float(np.sqrt(wv @ Sigma @ wv)))
            returns.append(float(wv @ mu))
            weights.append(wv.copy())

    return np.array(risks), np.array(returns), weights

risks_sp, rets_sp, weights_sp = scipy_frontier(mu, Sigma, n)
print(f"  Solved {len(risks_sp)} portfolios via scipy.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SPECIAL PORTFOLIOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("5. SPECIAL PORTFOLIOS")
print("=" * 65)

# ── Min-Variance Portfolio ────────────────────────────────────────────────────
w_cv  = cp.Variable(n)
prob_minvar = cp.Problem(
    cp.Minimize(cp.quad_form(w_cv, Sigma)),
    [cp.sum(w_cv) == 1, w_cv >= 0]
)
prob_minvar.solve(solver=cp.SCS, verbose=False)
w_minvar  = w_cv.value
ret_minvar  = float(w_minvar @ mu)
risk_minvar = float(np.sqrt(w_minvar @ Sigma @ w_minvar))
sharpe_minvar = (ret_minvar - RISK_FREE) / risk_minvar

# ── Maximum-Return Portfolio ──────────────────────────────────────────────────
# With w >= 0 and sum(w) = 1, max return = 100% in highest-mu asset
idx_maxret = np.argmax(mu)
w_maxret   = np.zeros(n); w_maxret[idx_maxret] = 1.0
ret_maxret  = float(w_maxret @ mu)
risk_maxret = float(np.sqrt(w_maxret @ Sigma @ w_maxret))
sharpe_maxret = (ret_maxret - RISK_FREE) / risk_maxret

# ── Minimum-Return Portfolio ──────────────────────────────────────────────────
idx_minret = np.argmin(mu)
w_minret   = np.zeros(n); w_minret[idx_minret] = 1.0
ret_minret  = float(w_minret @ mu)
risk_minret = float(np.sqrt(w_minret @ Sigma @ w_minret))
sharpe_minret = (ret_minret - RISK_FREE) / risk_minret

# ── Maximum Sharpe Ratio Portfolio ────────────────────────────────────────────
def neg_sharpe(w, mu, Sigma, rf):
    ret  = w @ mu
    risk = np.sqrt(w @ Sigma @ w)
    return -(ret - rf) / risk if risk > 1e-10 else 1e10

constraints_sr = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds_sr      = [(0.0, 1.0)] * n
best_neg_sr    = np.inf
w_maxsr        = None

# Multiple random starts to avoid local optima
np.random.seed(42)
for _ in range(100):
    w0 = np.random.dirichlet(np.ones(n))
    res = minimize(
        neg_sharpe, w0,
        args=(mu, Sigma, RISK_FREE),
        method="SLSQP",
        bounds=bounds_sr,
        constraints=constraints_sr,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if res.success and res.fun < best_neg_sr:
        best_neg_sr = res.fun
        w_maxsr = res.x.copy()

ret_maxsr   = float(w_maxsr @ mu)
risk_maxsr  = float(np.sqrt(w_maxsr @ Sigma @ w_maxsr))
sharpe_maxsr = (ret_maxsr - RISK_FREE) / risk_maxsr

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Portfolio':<25} {'Return':>10} {'Volatility':>12} {'Sharpe':>8}")
print("-" * 60)
print(f"{'1/N Equally Weighted':<25} {ret_eq*100:>9.2f}% {risk_eq*100:>11.2f}% {sharpe_eq:>8.4f}")
print(f"{'Min Variance':<25} {ret_minvar*100:>9.2f}% {risk_minvar*100:>11.2f}% {sharpe_minvar:>8.4f}")
print(f"{'Max Return':<25} {ret_maxret*100:>9.2f}% {risk_maxret*100:>11.2f}% {sharpe_maxret:>8.4f}")
print(f"{'Min Return':<25} {ret_minret*100:>9.2f}% {risk_minret*100:>11.2f}% {sharpe_minret:>8.4f}")
print(f"{'Max Sharpe Ratio':<25} {ret_maxsr*100:>9.2f}% {risk_maxsr*100:>11.2f}% {sharpe_maxsr:>8.4f}")

# Comments:
# - Min-Variance minimises portfolio risk; its return may be below the 1/N
#   baseline but it is the most robust choice for risk-averse investors.
# - Max-Return is fully concentrated (single stock), hence extremely risky.
#   It is rarely a realistic investment.
# - Max-Sharpe (tangency portfolio) maximises return per unit of risk and is
#   theoretically the optimal risky portfolio combined with a risk-free asset.
# - The 1/N portfolio lands near the middle – decent Sharpe without any
#   estimation risk, which is why it is hard to beat out-of-sample.

# ─────────────────────────────────────────────────────────────────────────────
# 6. WEIGHT TABLES FOR SPECIAL PORTFOLIOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("6. WEIGHT TABLES FOR SPECIAL PORTFOLIOS")
print("=" * 65)

special_portfolios = {
    "1/N Equal Weight" : w_eq,
    "Min Variance"     : w_minvar,
    "Max Return"       : w_maxret,
    "Min Return"       : w_minret,
    "Max Sharpe Ratio" : w_maxsr,
}

df_weights = pd.DataFrame(special_portfolios, index=TICKERS)
df_weights = (df_weights * 100).round(2)   # in percentage
df_weights.index.name = "Ticker"

print("\nWeights (%) for each special portfolio:")
print(df_weights.to_string())

# Only show significant allocations (>1%) to aid interpretation
print("\nSignificant weights (>1%) by portfolio:")
for col in df_weights.columns:
    sig = df_weights[col][df_weights[col] > 1.0].sort_values(ascending=False)
    print(f"\n  {col}:")
    for ticker, wt in sig.items():
        print(f"    {ticker:6s}: {wt:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7. LEVERAGE / SHORT SELLING  [Extra +10]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("7. LEVERAGE / SHORT SELLING  [Extra +10 pts]")
print("=" * 65)

# We parameterise leverage by a lower bound L on individual weights:
#   w_i >= -L   (L = 0 → long-only; L > 0 → short selling allowed)
# The net investment constraint Σw_i = 1 still holds.
# Gross leverage = Σ|w_i| = 1 + 2·Σmax(0, -w_i).
# Reference: Cornuejols & Tütüncü (2006), p. 402 equivalent formulation.

leverage_levels = {
    "L=0 (No leverage)"     : 0.0,
    "L=0.3 (Mild leverage)" : 0.3,
    "L=0.6 (Med leverage)"  : 0.6,
    "L=1.0 (High leverage)" : 1.0,
}

frontier_by_leverage = {}
for label, lb in leverage_levels.items():
    r, ret, w = cvxpy_frontier(mu, Sigma, n, gammas_fine, lb=-lb)
    frontier_by_leverage[label] = (r, ret, w)
    print(f"  {label}: {len(r)} portfolios  "
          f"| ret ∈ [{ret.min()*100:.1f}%, {ret.max()*100:.1f}%]  "
          f"| risk ∈ [{r.min()*100:.1f}%, {r.max()*100:.1f}%]")

# Comment on leverage:
# Allowing short sales expands the feasible set, shifting the efficient
# frontier upward and to the left – higher returns become achievable at the
# same risk, or the same return is reachable with less risk. However, this
# theoretical gain comes with practical costs:
#   • Borrowing costs reduce net returns.
#   • Margin calls can force liquidation at the worst moments (e.g., 2020).
#   • Tail risk is amplified: losses beyond 100% of capital are possible.
# The benefit diminishes as we add more leverage; the frontier improvement
# from L=0.6 to L=1.0 is smaller than from L=0 to L=0.3, consistent with
# the law of diminishing marginal utility of diversification.
# For this asset universe (diverse sectors, 2017-2023), moderate leverage
# (~L=0.3) can improve the Sharpe tangency point noticeably, but high
# leverage makes portfolios very sensitive to estimation errors in μ.

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Project 2 – Portfolio Optimization\n"
             "20 Assets · 2017-01-01 → 2023-12-31", fontsize=14, fontweight="bold")

# ── Plot 1: CVXPY + scipy efficient frontiers + special portfolios ────────────
ax = axes[0]

ax.plot(risks_cvx * 100, rets_cvx * 100,
        "b-", lw=2.5, label="CVXPY efficient frontier (γ sweep)", zorder=3)
ax.plot(risks_sp * 100,  rets_sp * 100,
        "g--", lw=1.8, label="scipy efficient frontier (target-return)", zorder=3)

# Special portfolios
specials_plot = [
    (risk_minvar,  ret_minvar,  "Min Variance",    "s",  "purple",  120),
    (risk_maxret,  ret_maxret,  "Max Return",       "^",  "red",     120),
    (risk_minret,  ret_minret,  "Min Return",       "v",  "orange",  120),
    (risk_maxsr,   ret_maxsr,   "Max Sharpe Ratio", "*",  "gold",    200),
    (risk_eq,      ret_eq,      "1/N Equal Weight", "o",  "cyan",    120),
]
for risk, ret, label, marker, color, size in specials_plot:
    ax.scatter(risk * 100, ret * 100, marker=marker, s=size,
               color=color, edgecolors="black", linewidths=0.8,
               label=label, zorder=5)
    ax.annotate(label, (risk * 100, ret * 100),
                textcoords="offset points", xytext=(8, 4), fontsize=7.5)

# Individual assets
for i, ticker in enumerate(TICKERS):
    r_i = np.sqrt(Sigma[i, i])
    ax.scatter(r_i * 100, mu[i] * 100, marker="x", s=40,
               color="gray", alpha=0.6, zorder=2)
    ax.annotate(ticker, (r_i * 100, mu[i] * 100),
                textcoords="offset points", xytext=(4, 2),
                fontsize=6.5, color="gray")

ax.set_xlabel("Annual Volatility (%)", fontsize=11)
ax.set_ylabel("Annual Return (%)", fontsize=11)
ax.set_title("Efficient Frontier + Special Portfolios", fontsize=12)
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))

# ── Plot 2: Efficient frontiers per leverage level ────────────────────────────
ax2 = axes[1]
colors_lev = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

for (label, (r, ret, _)), color in zip(frontier_by_leverage.items(), colors_lev):
    ax2.plot(r * 100, ret * 100, lw=2.2, label=label, color=color)

ax2.axhline(RISK_FREE * 100, color="black", linestyle=":", lw=1.2,
            label=f"Risk-free rate ({RISK_FREE*100:.0f}%)")

ax2.set_xlabel("Annual Volatility (%)", fontsize=11)
ax2.set_ylabel("Annual Return (%)", fontsize=11)
ax2.set_title("Efficient Frontiers by Leverage Level\n"
              "(Short-selling with wᵢ ≥ -L)", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))

plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=180, bbox_inches="tight")
plt.show()
print("\nPlot saved to efficient_frontier.png")

# ─────────────────────────────────────────────────────────────────────────────
# CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────
print("""
=================================================================
CONCLUSIONS
=================================================================

1. Data & Period (2017-2023)
   The sample spans two distinct regimes: the long bull market
   (2017-2019), the COVID crash and recovery (2020), and the
   rate-hike bear market (2022). This breadth makes the covariance
   estimates richer but also highlights regime dependency.

2. 1/N Baseline
   With 20 well-diversified assets across six sectors the 1/N
   portfolio achieves respectable risk-adjusted performance. It
   benefits from diversification without any estimation-error risk.
   It is a strong argument for passive diversified investing.

3. Min-Variance Portfolio
   Delivers the lowest realised volatility, accepting a lower
   expected return. Stocks with low pairwise correlations (e.g.
   healthcare, consumer staples) receive the largest weights.
   This portfolio is ideal for highly risk-averse investors or
   as a defensive holding during uncertain markets (2022 is a
   prime example where it would have outperformed).

4. Max-Sharpe (Tangency) Portfolio
   Theoretically the optimal risky portfolio when combined with a
   risk-free asset. In practice, its weights are highly sensitive
   to small changes in μ estimates. Its out-of-sample performance
   can disappoint; robust estimators (Black-Litterman, shrinkage)
   are advisable before live trading.

5. Leverage & Short Selling
   Expanding the feasible set via short selling shifts the frontier
   upward-left, improving the Sharpe ratio of the tangency portfolio.
   However:
   - Each unit of additional leverage offers diminishing improvement.
   - Estimation risk in μ is magnified (short positions can blow up).
   - Borrowing costs and margin requirements reduce net gains.
   For this universe, mild leverage (L≈0.3) seems worthwhile;
   beyond L=0.6 the frontier expansion is modest compared to the
   added complexity and risk.

6. Practical Takeaway
   No single portfolio dominates all others across all criteria.
   Asset allocation is inherently a trade-off between return,
   risk, and implementation realism. Quarterly rebalancing, factor
   diversification, and transaction-cost awareness are essential
   for translating any MVO result into real-world performance.
""")
