import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime

# ---------------------------- Configuration ---------------------------- #
TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY"]
START_DATE = "2020-01-01"
END_DATE = "2024-07-31"
DATA_DIR = "data"
PLOTS_DIR = "plots"
RISK_FREE_RATE = 0.01

print("📈 STOCK MARKET ANALYSIS")
print("=" * 50)

# ---------------------------- Setup Directories ---------------------------- #
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------- Data Download (One ticker at a time) ---------------------------- #
data = {}
successful_downloads = []
failed_downloads = []

for ticker in TICKERS:
    print(f"📥 Downloading {ticker}...")

    try:
        # Download one ticker at a time to avoid MultiIndex issues
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=START_DATE, end=END_DATE, auto_adjust=True)

        if df.empty:
            print(f"❌ No data for {ticker}")
            failed_downloads.append(f"{ticker}: No data returned")
            continue

        if 'Close' not in df.columns:
            print(f"❌ No Close column for {ticker}")
            failed_downloads.append(f"{ticker}: No Close column")
            continue

        close_prices = df['Close'].dropna()

        if len(close_prices) < 100:
            print(f"❌ Insufficient data for {ticker} ({len(close_prices)} points)")
            failed_downloads.append(f"{ticker}: Only {len(close_prices)} data points")
            continue

        # Store the close prices
        data[ticker] = close_prices
        successful_downloads.append(f"{ticker}: {len(close_prices)} data points")

        # Save raw data
        df.to_csv(f"{DATA_DIR}/{ticker}.csv")
        print(
            f"✅ {ticker}: {len(close_prices)} data points from {close_prices.index.min().date()} to {close_prices.index.max().date()}")

    except Exception as e:
        print(f"❌ Error downloading {ticker}: {str(e)}")
        failed_downloads.append(f"{ticker}: {str(e)}")

print(f"\n📊 Download Summary:")
print(f"✅ Successful: {len(successful_downloads)}")
print(f"❌ Failed: {len(failed_downloads)}")

for success in successful_downloads:
    print(f"  ✅ {success}")
for failure in failed_downloads:
    print(f"  ❌ {failure}")

if len(data) < 2:
    print(f"\n🔄 Trying backup tickers...")
    backup_tickers = ["TSLA", "AMZN", "NVDA", "META"]

    for backup_ticker in backup_tickers:
        if len(data) >= 4:
            break

        try:
            print(f"📥 Downloading backup ticker {backup_ticker}...")
            ticker_obj = yf.Ticker(backup_ticker)
            df = ticker_obj.history(start="2022-01-01", end=END_DATE, auto_adjust=True)

            if not df.empty and 'Close' in df.columns:
                close_prices = df['Close'].dropna()
                if len(close_prices) >= 100:
                    data[backup_ticker] = close_prices
                    df.to_csv(f"{DATA_DIR}/{backup_ticker}.csv")
                    print(f"✅ {backup_ticker}: {len(close_prices)} data points")
                else:
                    print(f"❌ {backup_ticker}: Insufficient data ({len(close_prices)} points)")
            else:
                print(f"❌ {backup_ticker}: No data or Close column")

        except Exception as e:
            print(f"❌ Error with backup ticker {backup_ticker}: {str(e)}")

if len(data) < 2:
    raise ValueError(f"🚫 Only {len(data)} ticker(s) downloaded successfully. Need at least 2 for analysis.")

print(f"\n🎯 Proceeding with analysis using {len(data)} tickers: {list(data.keys())}")

# ---------------------------- Create Combined DataFrame ---------------------------- #
# Align all data to common date range
combined_df = pd.DataFrame(data)

# Remove any rows where we don't have data for all tickers
print(f"📊 Before alignment: {combined_df.shape}")
combined_df = combined_df.dropna()
print(f"📊 After alignment: {combined_df.shape}")

if combined_df.empty:
    # If no common dates, use forward fill
    combined_df = pd.DataFrame(data)
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    combined_df = combined_df.dropna()
    print(f"📊 After forward fill: {combined_df.shape}")

if combined_df.empty or len(combined_df) < 50:
    raise ValueError("🚫 Not enough overlapping data between tickers for analysis.")

# Calculate returns
returns_df = combined_df.pct_change().dropna()
print(f"📊 Returns data shape: {returns_df.shape}")


# ---------------------------- Calculations ---------------------------- #
def calculate_metrics(returns_df, prices_df, risk_free_rate=0.01):
    """Calculate key financial metrics"""

    metrics = {}

    for ticker in returns_df.columns:
        ticker_returns = returns_df[ticker]

        # Annualized volatility
        volatility = ticker_returns.std() * np.sqrt(252)

        # Annualized return
        annual_return = ticker_returns.mean() * 252

        # Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / volatility

        # Max drawdown
        price_series = prices_df[ticker]
        rolling_max = price_series.expanding().max()
        drawdown = (price_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        metrics[ticker] = {
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown
        }

    return pd.DataFrame(metrics).T


# Calculate metrics for all tickers
metrics_df = calculate_metrics(returns_df, combined_df, RISK_FREE_RATE)

# Calculate betas (if SPY is available)
betas = {}
if 'SPY' in returns_df.columns:
    spy_returns = returns_df['SPY']
    for ticker in returns_df.columns:
        if ticker != 'SPY':
            try:
                # Simple correlation-based beta calculation
                covariance = np.cov(returns_df[ticker], spy_returns)[0][1]
                spy_variance = np.var(spy_returns)
                beta = covariance / spy_variance
                betas[ticker] = beta
            except:
                betas[ticker] = np.nan

# Add betas to metrics
if betas:
    metrics_df['Beta_vs_SPY'] = pd.Series(betas)

# Correlation matrix
correlation_matrix = returns_df.corr()

print(f"\n📈 ANALYSIS RESULTS")
print("=" * 50)
print("Key Metrics:")
print(metrics_df.round(3))
print(f"\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# ---------------------------- Visualizations ---------------------------- #

# 1. Price Performance (Normalized to start at 100)
plt.figure(figsize=(12, 8))
normalized_prices = combined_df / combined_df.iloc[0] * 100
for ticker in normalized_prices.columns:
    plt.plot(normalized_prices.index, normalized_prices[ticker], label=ticker, linewidth=2)

plt.title('Normalized Price Performance (Base = 100)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Normalized Price', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/price_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Risk-Return Scatter Plot
plt.figure(figsize=(10, 8))
for ticker in metrics_df.index:
    plt.scatter(metrics_df.loc[ticker, 'Volatility'],
                metrics_df.loc[ticker, 'Annual_Return'],
                s=100, label=ticker)
    plt.annotate(ticker,
                 (metrics_df.loc[ticker, 'Volatility'], metrics_df.loc[ticker, 'Annual_Return']),
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('Volatility (Annual)', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.title('Risk vs Return Profile', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/risk_return.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Returns Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Volatility Comparison
plt.figure(figsize=(10, 6))
volatilities = metrics_df['Volatility'].sort_values()
colors = plt.cm.viridis(np.linspace(0, 1, len(volatilities)))
bars = plt.bar(range(len(volatilities)), volatilities, color=colors)
plt.xlabel('Stocks', fontsize=12)
plt.ylabel('Annualized Volatility', fontsize=12)
plt.title('Volatility Comparison', fontsize=16, fontweight='bold')
plt.xticks(range(len(volatilities)), volatilities.index, rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/volatility_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------- Save Results ---------------------------- #
# Save all data
combined_df.to_csv(f'{DATA_DIR}/combined_prices.csv')
returns_df.to_csv(f'{DATA_DIR}/returns.csv')
metrics_df.to_csv(f'{DATA_DIR}/metrics.csv')
correlation_matrix.to_csv(f'{DATA_DIR}/correlation_matrix.csv')

# Save summary report
with open(f'{DATA_DIR}/analysis_summary.txt', 'w') as f:
    f.write("STOCK MARKET ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Data Period: {combined_df.index.min().date()} to {combined_df.index.max().date()}\n")
    f.write(f"Trading Days: {len(combined_df)}\n")
    f.write(f"Stocks Analyzed: {', '.join(combined_df.columns)}\n\n")

    f.write("KEY METRICS:\n")
    f.write("-" * 30 + "\n")
    for ticker in metrics_df.index:
        f.write(f"{ticker}:\n")
        f.write(f"  Annual Return: {metrics_df.loc[ticker, 'Annual_Return']:.1%}\n")
        f.write(f"  Volatility: {metrics_df.loc[ticker, 'Volatility']:.1%}\n")
        f.write(f"  Sharpe Ratio: {metrics_df.loc[ticker, 'Sharpe_Ratio']:.2f}\n")
        f.write(f"  Max Drawdown: {metrics_df.loc[ticker, 'Max_Drawdown']:.1%}\n")
        if 'Beta_vs_SPY' in metrics_df.columns and not pd.isna(metrics_df.loc[ticker, 'Beta_vs_SPY']):
            f.write(f"  Beta (vs SPY): {metrics_df.loc[ticker, 'Beta_vs_SPY']:.2f}\n")
        f.write("\n")

print(f"\n✅ ANALYSIS COMPLETE!")
print("=" * 50)
print(f"📁 Data files saved in: {DATA_DIR}/")
print(f"📊 Charts saved in: {PLOTS_DIR}/")
print(f"📈 Analyzed {len(combined_df.columns)} stocks over {len(combined_df)} trading days")
print(f"📅 Period: {combined_df.index.min().date()} to {combined_df.index.max().date()}")
print("=" * 50)