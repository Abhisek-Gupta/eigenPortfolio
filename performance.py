import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_performance_metrics(port_val, bench_val, risk_free_rate=0.06, periods_per_year=52):
    """
    Calculates institutional-grade performance metrics for a portfolio vs benchmark.
    
    Parameters:
    port_val (pd.Series): Portfolio equity curve over time.
    bench_val (pd.Series): Benchmark equity curve over time.
    risk_free_rate (float): Annual risk-free rate (default 6.0% for Indian markets).
    periods_per_year (int): Trading periods in a year (252 for daily, 52 for weekly, 12 for monthly).
    
    Returns:
    metrics_df (pd.DataFrame): Formatted performance table.
    port_dd, bench_dd, port_ret, bench_ret: Series needed for plotting.
    """
    # 1. Ensure absolute date alignment
    df = pd.concat([port_val, bench_val], axis=1).dropna()
    df.columns = ['Portfolio', 'Benchmark']
    
    # Calculate periodic returns
    returns = df.pct_change().dropna()
    port_ret = returns['Portfolio']
    bench_ret = returns['Benchmark']
    
    # 2. Annualized Return (Geometric CAGR)
    total_periods = len(returns)
    ann_port_ret = (df['Portfolio'].iloc[-1] / df['Portfolio'].iloc[0]) ** (periods_per_year / total_periods) - 1
    ann_bench_ret = (df['Benchmark'].iloc[-1] / df['Benchmark'].iloc[0]) ** (periods_per_year / total_periods) - 1
    
    # 3. Annualized Volatility
    ann_port_vol = port_ret.std() * np.sqrt(periods_per_year)
    ann_bench_vol = bench_ret.std() * np.sqrt(periods_per_year)
    
    # 4. Sharpe Ratio
    port_sharpe = (ann_port_ret - risk_free_rate) / ann_port_vol
    bench_sharpe = (ann_bench_ret - risk_free_rate) / ann_bench_vol
    
    # 5. Sortino Ratio (Strict Downside Deviation)
    port_downside_sq = np.where(port_ret < 0, port_ret**2, 0)
    port_downside_dev = np.sqrt(np.mean(port_downside_sq)) * np.sqrt(periods_per_year)
    port_sortino = (ann_port_ret - risk_free_rate) / port_downside_dev if port_downside_dev > 0 else np.nan
    
    bench_downside_sq = np.where(bench_ret < 0, bench_ret**2, 0)
    bench_downside_dev = np.sqrt(np.mean(bench_downside_sq)) * np.sqrt(periods_per_year)
    bench_sortino = (ann_bench_ret - risk_free_rate) / bench_downside_dev if bench_downside_dev > 0 else np.nan
    
    # 6. Maximum Drawdown
    port_dd = (df['Portfolio'] - df['Portfolio'].cummax()) / df['Portfolio'].cummax()
    bench_dd = (df['Benchmark'] - df['Benchmark'].cummax()) / df['Benchmark'].cummax()
    max_port_dd = port_dd.min()
    max_bench_dd = bench_dd.min()
    
    # 7. Calmar Ratio
    port_calmar = ann_port_ret / abs(max_port_dd) if max_port_dd < 0 else np.nan
    bench_calmar = ann_bench_ret / abs(max_bench_dd) if max_bench_dd < 0 else np.nan
    
    # 8. Beta & Jensen's Alpha
    cov_matrix = np.cov(port_ret, bench_ret)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # CAPM Expected Return = Rf + Beta * (Market Return - Rf)
    expected_return = risk_free_rate + beta * (ann_bench_ret - risk_free_rate)
    port_alpha = ann_port_ret - expected_return
    
    # 9. Format Results Table
    metrics_df = pd.DataFrame({
        'Portfolio': [
            f"{ann_port_ret*100:.2f}%", 
            f"{ann_port_vol*100:.2f}%", 
            f"{port_sharpe:.2f}", 
            f"{port_sortino:.2f}", 
            f"{max_port_dd*100:.2f}%", 
            f"{port_calmar:.2f}", 
            f"{beta:.2f}", 
            f"{port_alpha*100:.2f}%"
        ],
        'Benchmark': [
            f"{ann_bench_ret*100:.2f}%", 
            f"{ann_bench_vol*100:.2f}%", 
            f"{bench_sharpe:.2f}", 
            f"{bench_sortino:.2f}", 
            f"{max_bench_dd*100:.2f}%", 
            f"{bench_calmar:.2f}", 
            "1.00", 
            "0.00%"
        ]
    }, index=[
        'Annualized Return (CAGR)', 
        'Annualized Volatility', 
        'Sharpe Ratio', 
        'Sortino Ratio', 
        'Max Drawdown', 
        'Calmar Ratio', 
        'Beta (Market Risk)', 
        "Jensen's Alpha"
    ])
    
    return metrics_df, port_dd, bench_dd, port_ret, bench_ret


def plot_tearsheet(port_val, bench_val, port_dd, bench_dd, port_ret, bench_ret, title="Eigen-Portfolio Backtest", rolling_window=26):
    """
    Generates a 3-part quantitative tearsheet: Normalized Equity Curve, Drawdowns, and Rolling Beta.
    """
    # Normalize initial equity to exactly 100 for clean visual comparison
    port_norm = (port_val / port_val.iloc[0]) * 100
    bench_norm = (bench_val / bench_val.iloc[0]) * 100
    
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.3)
    
    # --- Plot 1: Cumulative Returns ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(port_norm.index, port_norm, label='PCA Portfolio', color='#1f77b4', linewidth=2.5)
    ax1.plot(bench_norm.index, bench_norm, label='Benchmark', color='#4a4a4a', linewidth=1.5, alpha=0.8)
    ax1.set_title(f'{title} (Normalized to ₹100 Base)', fontsize=14, fontweight='bold', loc='left')
    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, fontsize=11)
    
    # --- Plot 2: Underwater Plot (Drawdowns) ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(port_dd.index, port_dd * 100, 0, label='Portfolio Drawdown', color='#d62728', alpha=0.3)
    ax2.plot(bench_dd.index, bench_dd * 100, label='Benchmark Drawdown', color='black', linewidth=1.2, alpha=0.7)
    ax2.set_title('Underwater Plot (Capital Drawdowns)', fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylabel('Drop from Peak (%)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=10)
    
    # --- Plot 3: Rolling Beta ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    rolling_cov = port_ret.rolling(window=rolling_window).cov(bench_ret)
    rolling_var = bench_ret.rolling(window=rolling_window).var()
    rolling_beta = rolling_cov / rolling_var
    
    ax3.plot(rolling_beta.index, rolling_beta, color='#2ca02c', linewidth=2, label=f'Rolling {rolling_window}-Period Beta')
    ax3.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Market Neutral (1.0)')
    ax3.set_title(f'Rolling Systemic Risk Exposure ({rolling_window} Periods)', fontsize=12, fontweight='bold', loc='left')
    ax3.set_ylabel('Beta', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='lower right', fontsize=10)
    
    plt.show()