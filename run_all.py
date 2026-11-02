"""
Retail Sales & Demand Analytics - end-to-end pipeline (run_all.py)
Reproduces 01_cleaning.ipynb, 02_eda.ipynb and 03_models.ipynb in order.
"""

import math
import os
import warnings

import matplotlib
matplotlib.use('Agg')  # headless backend so figures save without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')



# =========================================================================
# PHASE 1 & 2: DATA CLEANING AND FEATURE ENGINEERING (01_cleaning)
# =========================================================================

# -------------------------------------------------------------------------
# Step 1: Loading the Excel file and inspect row counts
# -------------------------------------------------------------------------


# File location
file_name = 'Brainpals_sales_analytics.xlsx'        # Main Excel file

# Load every sheet into its own DataFrame
stores = pd.read_excel(file_name, sheet_name='Stores')
products = pd.read_excel(file_name, sheet_name='Products')
weekly_sales = pd.read_excel(file_name, sheet_name='Weekly_Sales')
external = pd.read_excel(file_name, sheet_name='External_Weekly')
customers = pd.read_excel(file_name, sheet_name='Customers')

# Display how many rows and columns each sheet has
print("Stores:        ", stores.shape)
print("Products:      ", products.shape)
print("Weekly Sales:  ", weekly_sales.shape)
print("External:      ", external.shape)
print("Customers:     ", customers.shape)

# -------------------------------------------------------------------------
# Step 2: Finding duplicates and inconsistent labels
# -------------------------------------------------------------------------

# Check how many exact duplicate rows exist in Weekly_Sales
sales_duplicates = weekly_sales.duplicated().sum()
print("Duplicate rows in Weekly_Sales:", sales_duplicates)

# Inspect category spelling and whitespace in the Products table
print("\nUnique product categories before cleaning:")
print(products['category'].tolist())

# -------------------------------------------------------------------------
# Step 3: Removing duplicates and cleaning category names
# -------------------------------------------------------------------------

# 1. Drop duplicate rows from Weekly_Sales
weekly_sales = weekly_sales.drop_duplicates()
print("Weekly_Sales rows after dropping duplicates:", len(weekly_sales))

# 2. Clean category names: strip extra spaces and make Title Case
products['category'] = products['category'].str.strip().str.title()
print("\nClean product categories:")
print(products['category'].unique())

# -------------------------------------------------------------------------
# Step 4: Handling Negative and Missing Sales
# -------------------------------------------------------------------------

# 1. Identify rows with negative or missing sales
invalid_mask = (weekly_sales['units_sold'] <= 0) | (weekly_sales['units_sold'].isnull())
print(f"Total invalid rows to fix: {invalid_mask.sum()}")

# 2. Find median positive sales for each store and product pair
positive_sales = weekly_sales['units_sold'].where(weekly_sales['units_sold'] > 0)
typical_sales = positive_sales.groupby([weekly_sales['store_id'], weekly_sales['product_id']]).transform('median')

# 3. Replace invalid sales with typical sales using .mask()
weekly_sales['units_sold'] = weekly_sales['units_sold'].mask(invalid_mask, typical_sales)

# 4. Confirm sales are clean and positive
print(f"Minimum units_sold after fix: {weekly_sales['units_sold'].min()}")
print(f"Missing units_sold after fix: {weekly_sales['units_sold'].isnull().sum()}")

# -------------------------------------------------------------------------
# Step 5: Detecting Outliers Using IQR Rule and Per-Product Z-Score
# -------------------------------------------------------------------------

# 1. Calculate IQR fences per product
q25 = weekly_sales.groupby('product_id')['units_sold'].transform('quantile', 0.25)
q75 = weekly_sales.groupby('product_id')['units_sold'].transform('quantile', 0.75)
iqr = q75 - q25

lower_bound = q25 - (1.5 * iqr)
upper_bound = q75 + (1.5 * iqr)

# Flag IQR outliers (outside fences)
weekly_sales['outlier_iqr'] = (weekly_sales['units_sold'] < lower_bound) | (weekly_sales['units_sold'] > upper_bound)

# 2. Calculate Z-scores per product
mean_units = weekly_sales.groupby('product_id')['units_sold'].transform('mean')
std_units = weekly_sales.groupby('product_id')['units_sold'].transform('std')

# Flag Z-score outliers (more than 3 standard deviations away)
weekly_sales['outlier_zscore'] = ((weekly_sales['units_sold'] - mean_units) / std_units).abs() > 3

# 3. Compare results
iqr_total = weekly_sales['outlier_iqr'].sum()
z_total = weekly_sales['outlier_zscore'].sum()
overlap = (weekly_sales['outlier_iqr'] & weekly_sales['outlier_zscore']).sum()

print(f"Rows flagged by IQR rule:         {iqr_total}")
print(f"Rows flagged by Z-score (|z| > 3): {z_total}")
print(f"Rows flagged by both methods:     {overlap}")

# -------------------------------------------------------------------------
# Step 6: Impute Missing discount_pct from promo_flag and Price Logic
# -------------------------------------------------------------------------

# 1. Count missing values before fix
print(f"Missing discount_pct before: {weekly_sales['discount_pct'].isnull().sum()}")

# 2. Calculate regular undiscounted shelf price
shelf_price = weekly_sales['product_id'].map(products.set_index('product_id')['list_price_ngn']) * (
    weekly_sales['week_start'].map(external.set_index('week_start')['cpi_index']) / 100.0
)

# 3. Calculate implied discount ratio
raw_ratio = 1.0 - (weekly_sales['unit_price_ngn'] / shelf_price)

# 4. Calculate promo discount (snap to nearest 0.05 and clip between 5% and 30%)
promo_disc = (raw_ratio / 0.05).round() * 0.05
promo_disc = promo_disc.clip(lower=0.05, upper=0.30)

# 5. Calculate non-promo discount (0.05 if marked down, else 0.00)
non_promo_disc = pd.Series(0.0, index=weekly_sales.index).mask(raw_ratio >= 0.05, 0.05)

# 6. Select discount based on promo_flag (use promo_disc if promo_flag == 1, else non_promo_disc)
derived_discount = non_promo_disc.mask(weekly_sales['promo_flag'] == 1, promo_disc)

# 7. Fill only the missing entries
weekly_sales['discount_pct'] = weekly_sales['discount_pct'].fillna(derived_discount)

# 8. Verify missing values are resolved
print(f"Missing discount_pct after:  {weekly_sales['discount_pct'].isnull().sum()}")
print("Unique discount values present:", sorted(weekly_sales['discount_pct'].unique().tolist()))

# -------------------------------------------------------------------------
# Step 7: Cleaning discount_pct Precision & Recomputing revenue_ngn
# -------------------------------------------------------------------------

# 1. Clean float precision jitter on discount_pct
weekly_sales['discount_pct'] = weekly_sales['discount_pct'].round(2)
print("Standardized discount tiers:", sorted(weekly_sales['discount_pct'].unique().tolist()))

# 2. Check mismatches and missing values before recalculation
expected_revenue = weekly_sales['units_sold'] * weekly_sales['unit_price_ngn']
mismatches_before = ((weekly_sales['revenue_ngn'] - expected_revenue).abs() > 0.01).sum()
missing_rev_before = weekly_sales['revenue_ngn'].isnull().sum()

print(f"\nMissing revenue records before:    {missing_rev_before}")
print(f"Mismatched revenue records before: {mismatches_before}")

# 3. Recompute revenue_ngn directly
weekly_sales['revenue_ngn'] = (weekly_sales['units_sold'] * weekly_sales['unit_price_ngn']).round(2)

# 4. Verify mathematical integrity across all 46,800 rows
mismatches_after = ((weekly_sales['revenue_ngn'] - (weekly_sales['units_sold'] * weekly_sales['unit_price_ngn'])).abs() > 0.01).sum()
missing_rev_after = weekly_sales['revenue_ngn'].isnull().sum()

print(f"\nMissing revenue records after:     {missing_rev_after}")
print(f"Mismatched revenue records after:  {mismatches_after}")
print(f"Revenue range: NGN {weekly_sales['revenue_ngn'].min():,.2f} to NGN {weekly_sales['revenue_ngn'].max():,.2f}")

# -------------------------------------------------------------------------
# Step 8: Impute Store Foot Traffic & Clean Customer Age Anomalies
# -------------------------------------------------------------------------

# 1. Impute missing foot_traffic_index in Stores using the store_type median
missing_traffic_before = stores['foot_traffic_index'].isnull().sum()
print(f"Stores with missing foot_traffic_index before: {missing_traffic_before}")

# Calculate median foot traffic for each store format (Express, Standard, Hypermarket)
type_traffic_median = stores.groupby('store_type')['foot_traffic_index'].transform('median')

# Fill missing values using the store_type median
stores['foot_traffic_index'] = stores['foot_traffic_index'].fillna(type_traffic_median)

missing_traffic_after = stores['foot_traffic_index'].isnull().sum()
print(f"Stores with missing foot_traffic_index after:  {missing_traffic_after}")

# 2. Flag impossible customer ages vs missing customer ages
customers['invalid_age_flag'] = (customers['age'] <= 0) | (customers['age'] > 100)
customers['missing_age_flag'] = customers['age'].isnull()

print(f"\nImpossible customer ages flagged (<= 0 or > 100): {customers['invalid_age_flag'].sum()}")
print(f"Missing customer ages flagged (NaN):               {customers['missing_age_flag'].sum()}")

# Set impossible ages to None so they do not distort demographic calculations
customers.loc[customers['invalid_age_flag'], 'age'] = None

# Check valid age summary statistics
valid_ages = customers['age'].dropna()
print(f"\nClean Customer Age Range: {valid_ages.min():.0f} to {valid_ages.max():.0f} years (Median: {valid_ages.median():.0f})")

# -------------------------------------------------------------------------
# Step 9: Lossless Table Merge (Weekly_Sales + Stores + Products + External)
# -------------------------------------------------------------------------

# 1. Capture initial row count
initial_rows = len(weekly_sales)

# 2. Sequential Left Joins
master_df = weekly_sales.merge(stores, on='store_id', how='left')
master_df = master_df.merge(products, on='product_id', how='left')
master_df = master_df.merge(external, on='week_start', how='left')

# 3. Verify row counts and column totals
final_rows = len(master_df)

print(f"Weekly sales rows before merge:  {initial_rows:,}")
print(f"Master dataset rows after merge: {final_rows:,}")
print(f"Total merged columns:             {master_df.shape[1]}")

# 4. Integrity assertion check
assert initial_rows == final_rows == 46800, "Row count mismatch: Data was dropped or duplicated during merge!"
print("Verification check passed: Exactly 46,800 panel rows preserved.")

# 5. Check if any missing values exist in the merged dataset
remaining_nulls = master_df.isnull().sum().sum()
print(f"Total remaining null values across master table: {remaining_nulls}")

# -------------------------------------------------------------------------
# Step 10: Real Prices, Price Ratios, and Log Transformations
# -------------------------------------------------------------------------

# 1. Deflate nominal prices using CPI index (base 100)
master_df['real_price'] = (master_df['unit_price_ngn'] / master_df['cpi_index']) * 100.0

# 2. Base period real list price equals list_price_ngn
master_df['real_list_price'] = master_df['list_price_ngn']

# 3. Compute price ratio
master_df['price_ratio'] = master_df['real_price'] / master_df['real_list_price']

# 4. Compute natural logarithms using math.log
master_df['log_units'] = master_df['units_sold'].apply(math.log)
master_df['log_price_ratio'] = master_df['price_ratio'].apply(math.log)

# Summary check
print("Feature Engineering Check (Prices & Logs):")
print(f"Price Ratio range: {master_df['price_ratio'].min():.2f} to {master_df['price_ratio'].max():.2f}")
print(f"Log Units range:   {master_df['log_units'].min():.2f} to {master_df['log_units'].max():.2f}")

# -------------------------------------------------------------------------
# Step 11: Calendar Features and Trigonometric Seasonal Terms
# -------------------------------------------------------------------------

# 1. Ensure week_start is datetime format
master_df['week_start'] = pd.to_datetime(master_df['week_start'])

# 2. Extract standard calendar parts
master_df['week_of_year'] = master_df['week_start'].dt.isocalendar().week.astype(int)  # 1 to 52
master_df['month'] = master_df['week_start'].dt.month                                   # 1 to 12
master_df['quarter'] = master_df['week_start'].dt.quarter                               # 1 to 4

# 3. Create cyclical Fourier terms using Python's built-in math module
master_df['sin_season'] = master_df['week_of_year'].apply(lambda w: math.sin(2 * math.pi * w / 52.0))
master_df['cos_season'] = master_df['week_of_year'].apply(lambda w: math.cos(2 * math.pi * w / 52.0))

# Check summary ranges
print("Calendar & Seasonality Features:")
print(f"Week of year range: {master_df['week_of_year'].min()} to {master_df['week_of_year'].max()}")
print(f"Month range:        {master_df['month'].min()} to {master_df['month'].max()}")
print(f"Quarter range:      {master_df['quarter'].min()} to {master_df['quarter'].max()}")
print(f"Sin term range:     {master_df['sin_season'].min():.2f} to {master_df['sin_season'].max():.2f}")
print(f"Cos term range:     {master_df['cos_season'].min():.2f} to {master_df['cos_season'].max():.2f}")

# -------------------------------------------------------------------------
# Step 12: Profit Margins & Store Productivity Metrics
# -------------------------------------------------------------------------

# 1. Total Cost and Gross Profit in NGN
total_cost = master_df['units_sold'] * master_df['unit_cost_ngn']
master_df['gross_profit'] = (master_df['revenue_ngn'] - total_cost).round(2)

# 2. Profit Margin Percentage (Gross Profit ÷ Revenue)
master_df['margin_pct'] = (master_df['gross_profit'] / master_df['revenue_ngn']).round(4)

# 3. Total store revenue over the entire period (aggregated by store)
store_total_rev = master_df.groupby('store_id')['revenue_ngn'].transform('sum')

# 4. Store Productivity metrics
master_df['sales_per_sqm'] = (store_total_rev / master_df['size_sqm']).round(2)
master_df['staff_productivity'] = (store_total_rev / master_df['staff_count']).round(2)

# Verification check
print("Margins and Productivity calculated:")
print(f"Mean Gross Margin:          {master_df['margin_pct'].mean() * 100:.2f}%")
print(f"Sales per sqm (range):      NGN {master_df['sales_per_sqm'].min():,.2f} to NGN {master_df['sales_per_sqm'].max():,.2f}")
print(f"Staff productivity (range): NGN {master_df['staff_productivity'].min():,.2f} to NGN {master_df['staff_productivity'].max():,.2f}")

# -------------------------------------------------------------------------
# Step 13: Time-Series 4-Week Lags and Rolling Means
# -------------------------------------------------------------------------

# 1. Sort chronologically by store, product, and week
master_df = master_df.sort_values(['store_id', 'product_id', 'week_start']).reset_index(drop=True)

# 2. Group by store and product pair
pair_group = master_df.groupby(['store_id', 'product_id'])['units_sold']

# 3. 4-week lag (sales volume from 4 weeks prior)
master_df['units_lag_4'] = pair_group.shift(4)

# 4. 4-week rolling mean
master_df['units_rolling_4'] = pair_group.transform(lambda s: s.rolling(4).mean())

# 5. Handle initial 4-week boundary nulls within each group
master_df['units_lag_4'] = master_df.groupby(['store_id', 'product_id'])['units_lag_4'].bfill()
master_df['units_rolling_4'] = master_df.groupby(['store_id', 'product_id'])['units_rolling_4'].bfill()

print("Time-series features created successfully:")
print(f"Missing lag-4 units:        {master_df['units_lag_4'].isnull().sum()}")
print(f"Missing rolling-4 units:    {master_df['units_rolling_4'].isnull().sum()}")

# -------------------------------------------------------------------------
# Step 14: Save Cleaned Master Table to CSV
# -------------------------------------------------------------------------

# Export to CSV without index column
master_df.to_csv('cleaned_master.csv', index=False)

print(f"cleaned_master.csv created successfully!")
print(f"Final shape: {master_df.shape[0]:,} rows and {master_df.shape[1]} columns.")



# =========================================================================
# PHASE 3: EXPLORATORY DATA ANALYSIS (02_eda)
# =========================================================================

# 1. Apply clean Seaborn styling across all figures
sns.set_theme(style='whitegrid')

# 2. Ensure destination folder exists
os.makedirs('figures', exist_ok=True)

# 3. Load the cleaned master dataset
df = pd.read_csv('cleaned_master.csv')
df['week_start'] = pd.to_datetime(df['week_start'])

print(f"Loaded master dataset: {df.shape[0]:,} rows and {df.shape[1]} columns.")

# -------------------------------------------------------------------------
# Figure 1: Chain-Level Weekly Revenue & Additive Decomposition
# -------------------------------------------------------------------------

# 1. Aggregate revenue at chain level per week (scaled to Millions of NGN)
weekly_rev = df.groupby('week_start')['revenue_ngn'].sum().reset_index()
weekly_rev['revenue_m_ngn'] = weekly_rev['revenue_ngn'] / 1e6
weekly_rev['rolling_12'] = weekly_rev['revenue_m_ngn'].rolling(window=12, min_periods=1).mean()

# 2. Decompose weekly revenue into trend, seasonality (52 weeks), and residual
decomp = seasonal_decompose(weekly_rev.set_index('week_start')['revenue_m_ngn'], model='additive', period=52)

# Build a tidy DataFrame for Seaborn
decomp_df = pd.DataFrame({
    'week_start': weekly_rev['week_start'],
    'Actual Revenue': weekly_rev['revenue_m_ngn'],
    '12-Week Rolling Mean': weekly_rev['rolling_12'],
    'Trend': decomp.trend.values,
    'Seasonality': decomp.seasonal.values,
    'Residual': decomp.resid.values
})

# 3. Plot 4 stacked subplots using Seaborn
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Panel 1: Observed Weekly Revenue and 12-Week Moving Average
sns.lineplot(data=weekly_rev, x='week_start', y='revenue_m_ngn', ax=axes[0], label='Actual Weekly Revenue', color='#1f77b4', lw=1.5)
sns.lineplot(data=weekly_rev, x='week_start', y='rolling_12', ax=axes[0], label='12-Week Rolling Mean', color='#ff7f0e', lw=2, linestyle='--')
axes[0].set_ylabel('Revenue (M NGN)')
axes[0].set_title('Figure 1: Chain-Level Weekly Revenue & Additive Decomposition', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left')

# Panel 2: Trend Component
sns.lineplot(data=decomp_df, x='week_start', y='Trend', ax=axes[1], color='#2ca02c', lw=2)
axes[1].set_ylabel('Trend (M NGN)')

# Panel 3: Seasonal Component
sns.lineplot(data=decomp_df, x='week_start', y='Seasonality', ax=axes[2], color='#9467bd', lw=1.5)
axes[2].set_ylabel('Seasonality (M NGN)')

# Panel 4: Residual Component
sns.scatterplot(data=decomp_df, x='week_start', y='Residual', ax=axes[3], color='#d62728', s=25)
axes[3].axhline(0, color='black', linestyle='--', linewidth=1)
axes[3].set_ylabel('Residual (M NGN)')
axes[3].set_xlabel('Week Start Date')

plt.tight_layout()

# 4. Save directly as 300 DPI PNG
plt.savefig('figures/fig1_revenue_ts.png', dpi=300)
print("Figure 1 saved to figures/fig1_revenue_ts.png (300 DPI)")

# -------------------------------------------------------------------------
# Figure 2: Correlation Heatmap (Numeric Store & Macro Variables)
# -------------------------------------------------------------------------

# 1. Define the store-level and external macro variables
store_macro_vars = [
    'size_sqm', 'staff_count', 'foot_traffic_index', 'competitors_2km', 
    'manager_tenure_yrs', 'avg_temp_c', 'rainfall_mm', 'cpi_index', 
    'fuel_price_ngn_per_l', 'holiday_week'
]

# 2. Compute Pearson correlation matrix
corr_matrix = df[store_macro_vars].corr()

# 3. Create the heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt='.2f', 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1, 
    center=0,
    linewidths=0.5, 
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)

plt.title('Figure 2: Correlation Heatmap of Store & Macroeconomic Variables', fontsize=12, pad=12)
plt.tight_layout()

# 4. Save at 300 DPI
plt.savefig('figures/fig2_correlation_heatmap.png', dpi=300)
print("Figure 2 saved to figures/fig2_correlation_heatmap.png")

# -------------------------------------------------------------------------
# Figure 3: Promo vs Non-Promo Units Sold by Category (Faceted)
# -------------------------------------------------------------------------

# 1. Create readable labels for promotional status
df['promo_label'] = df['promo_flag'].map({0: 'Non-Promo', 1: 'Promo'})

# 2. Build faceted boxplot across all 7 product categories
g = sns.catplot(
    data=df,
    x='promo_label',
    y='units_sold',
    hue='promo_label',       # Explicitly maps color palette to the x variable
    col='category',
    col_wrap=4,
    kind='box',
    showfliers=False,        # Suppresses outlier dots for clean interquartile comparison
    palette=['#4c72b0', '#dd8452'],
    legend=False,            # Suppresses redundant legend
    height=3.5,
    aspect=0.9,
    sharey=True
)

# 3. Apply titles and labels
g.set_axis_labels('', 'Weekly Units Sold')
g.set_titles(col_template='{col_name}', weight='bold')
g.fig.suptitle('Figure 3: Weekly Units Sold by Product Category (Promo vs Non-Promo)', fontsize=13, fontweight='bold', y=1.03)

# 4. Save figure directly at 300 DPI
g.savefig('figures/fig3_promo_by_category.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved cleanly to figures/fig3_promo_by_category.png")

# -------------------------------------------------------------------------
# Figure 4: Store Productivity Ranking (Revenue / Sqm vs. Revenue / Staff)
# -------------------------------------------------------------------------

# 1. Aggregate total revenue per store and calculate efficiency metrics
store_perf = df.groupby(['store_id', 'store_type', 'size_sqm', 'staff_count'])['revenue_ngn'].sum().reset_index()
store_perf['rev_per_sqm'] = store_perf['revenue_ngn'] / store_perf['size_sqm']
store_perf['rev_per_staff_m'] = (store_perf['revenue_ngn'] / store_perf['staff_count']) / 1e6

# 2. Sort descending by revenue per square metre
store_perf = store_perf.sort_values('rev_per_sqm', ascending=False).reset_index(drop=True)

# 3. Create side-by-side comparative horizontal barplots
fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)

# Left Panel: Store Ranking by Revenue per Sqm
sns.barplot(
    data=store_perf,
    y='store_id',
    x='rev_per_sqm',
    hue='store_type',
    dodge=False,
    ax=axes[0],
    palette='viridis'
)
axes[0].set_title('Stores Ranked by Revenue per Sqm (NGN)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Total Revenue / Size Sqm (NGN)')
axes[0].set_ylabel('Store ID')
axes[0].xaxis.set_major_formatter('{x:,.0f}')
axes[0].legend(title='Store Type', loc='lower right')

# Right Panel: Revenue per Staff Member (same store order)
sns.barplot(
    data=store_perf,
    y='store_id',
    x='rev_per_staff_m',
    hue='store_type',
    dodge=False,
    ax=axes[1],
    palette='viridis'
)
axes[1].set_title('Revenue per Staff Member (Million NGN)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Total Revenue / Staff Count (M NGN)')
axes[1].set_ylabel('')
axes[1].legend().remove()

plt.suptitle('Figure 4: Store Productivity Ranking (Revenue per Sqm vs. Revenue per Staff)', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()

# 4. Save directly as 300 DPI PNG
plt.savefig('figures/fig4_store_productivity.png', dpi=300, bbox_inches='tight')
print("Figure 4 saved to figures/fig4_store_productivity.png")

# -------------------------------------------------------------------------
# Figure 5: Log-Log Demand Curves by Product Category (Faceted)
# -------------------------------------------------------------------------

# 1. Build faceted scatter plot with linear regression fit per category
g = sns.lmplot(
    data=df,
    x='log_price_ratio',
    y='log_units',
    col='category',
    col_wrap=4,
    height=3.2,
    aspect=1.0,
    scatter_kws={'alpha': 0.15, 's': 10, 'color': '#1f77b4'},
    line_kws={'color': '#d62728', 'linewidth': 1.8}
)

# 2. Add clear academic axis titles
g.set_axis_labels('Log Price Ratio: log(P / P_list)', 'Log Units Sold: log(Q)')
g.set_titles(col_template='{col_name}', weight='bold')
g.fig.suptitle('Figure 5: Log-Log Demand Curves by Product Category', fontsize=13, fontweight='bold', y=1.03)

# 3. Save figure at 300 DPI
g.savefig('figures/fig5_log_price_vs_log_units.png', dpi=300, bbox_inches='tight')
print("Figure 5 saved to figures/fig5_log_price_vs_log_units.png")

# -------------------------------------------------------------------------
# Figure 6: Distribution of Weekly Units Sold by Store Type
# -------------------------------------------------------------------------

# 1. Create a 2-panel comparative layout
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Order formats from smallest to largest
store_order = ['Express', 'Standard', 'Hypermarket']

# 2. Left Panel: Boxplot focusing on the interquartile range (IQR)
sns.boxplot(
    data=df, 
    x='store_type', 
    y='units_sold', 
    order=store_order,
    color='#6baed6',        # Single clean colour avoids palette/hue warnings
    showfliers=False,       # Suppresses extreme fliers for clear quartile comparison
    ax=axes[0]
)
axes[0].set_title('Core Interquartile Distribution (Fliers Hidden)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Store Type')
axes[0].set_ylabel('Weekly Units Sold')

# 3. Right Panel: Violin plot with log-scale to reveal density shapes & tails
sns.violinplot(
    data=df, 
    x='store_type', 
    y='units_sold', 
    order=store_order,
    color='#6baed6',
    cut=0,                  # Bounds density strictly within actual observed range
    inner='quartile',       # Draws dashed lines for 25th, 50th (median), and 75th percentiles
    ax=axes[1]
)
axes[1].set_yscale('log')
axes[1].set_title('Density & Upper Tails (Log Scale)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Store Type')
axes[1].set_ylabel('Weekly Units Sold (Log Scale)')

plt.suptitle('Figure 6: Weekly Units Sold Distribution by Store Type', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# 4. Save figure at 300 DPI
plt.savefig('figures/fig6_units_by_store_type.png', dpi=300, bbox_inches='tight')
print("Figure 6 saved to figures/fig6_units_by_store_type.png")

# -------------------------------------------------------------------------
# Figure 7: Pearson vs. Spearman Correlation Matrix Comparison
# -------------------------------------------------------------------------

# 1. Select key analytical variables across sales, operations, and macro factors
corr_cols = [
    'units_sold', 'unit_price_ngn', 'discount_pct', 'promo_flag', 
    'size_sqm', 'foot_traffic_index', 'competitors_2km',
    'avg_temp_c', 'rainfall_mm', 'cpi_index', 'fuel_price_ngn_per_l'
]

# 2. Compute Pearson (linear) and Spearman (rank-order monotonic) correlations
pearson_corr = df[corr_cols].corr(method='pearson')
spearman_corr = df[corr_cols].corr(method='spearman')

# 3. Create a side-by-side comparative heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Pearson Linear Correlation
sns.heatmap(
    pearson_corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    center=0,
    cbar=False,
    ax=axes[0],
    square=True
)
axes[0].set_title('Pearson Correlation (Linear)', fontsize=11, fontweight='bold')

# Right: Spearman Monotonic Rank Correlation
sns.heatmap(
    spearman_corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    center=0,
    cbar=True,
    cbar_kws={'label': 'Correlation Coefficient'},
    ax=axes[1],
    square=True
)
axes[1].set_title('Spearman Correlation (Monotonic Rank)', fontsize=11, fontweight='bold')

plt.suptitle('Figure 7: Pearson vs. Spearman Correlation Matrix Comparison', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()

# 4. Save figure directly at 300 DPI
plt.savefig('figures/fig7_pearson_vs_spearman.png', dpi=300, bbox_inches='tight')
# 5. Print top discrepancies using pure Pandas (no NumPy required)
diff_corr = (pearson_corr - spearman_corr).abs().unstack()

# Keep unique pairs only (exclude diagonals and mirrored pairs)
unique_pairs = diff_corr[diff_corr.index.get_level_values(0) < diff_corr.index.get_level_values(1)]
top_5 = unique_pairs.sort_values(ascending=False).head(5)

print("\nTop 5 Discrepancies (|Pearson - Spearman|):")
for (v1, v2), diff_val in top_5.items():
    p_val = pearson_corr.loc[v1, v2]
    s_val = spearman_corr.loc[v1, v2]
    print(f"{v1} vs {v2}: Pearson = {p_val:.2f}, Spearman = {s_val:.2f} (Diff = {diff_val:.2f})")

# -------------------------------------------------------------------------
# Figure 8: Customer Total Spend vs. Visits Coloured by Loyalty Tier
# -------------------------------------------------------------------------

# 1. Load the Customers sheet
excel_file = 'Brainpals_sales_analytics.xlsx'
customers = pd.read_excel(excel_file, sheet_name='Customers')

# Identify visit, spend, and tier column names dynamically
spend_col = [c for c in customers.columns if 'spend' in c][0]
visits_col = [c for c in customers.columns if 'visit' in c][0]
tier_col = [c for c in customers.columns if 'tier' in c or 'loyalty' in c][0]

# Standardise tier text formatting
customers[tier_col] = customers[tier_col].astype(str).str.strip().str.title()

# 2. Scale spend to thousands of NGN (k NGN)
customers['total_spend_k_ngn'] = customers[spend_col] / 1e3

# 3. Dynamically extract the actual unique tiers present in your data
actual_tiers = sorted(customers[tier_col].unique().tolist())
print(f"Detected Loyalty Tiers: {actual_tiers}")

# 4. Create the scatter plot using Seaborn
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=customers,
    x=visits_col,
    y='total_spend_k_ngn',
    hue=tier_col,
    hue_order=actual_tiers,       # Matches the exact detected labels
    palette='tab10',              # Distinct, high-contrast colours
    alpha=0.75,
    s=55,
    edgecolor='w',                # Crisp white outline around each marker
    linewidth=0.5,
    ax=ax
)

# 5. Position legend cleanly
sns.move_legend(ax, "upper left", title='Loyalty Tier', frameon=True)

# 6. Add titles and axis labels
plt.title('Figure 8: Customer Lifetime Spend vs. Total Visits by Loyalty Tier', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Total Store Visits', fontsize=10)
plt.ylabel('Total Spend (Thousand NGN)', fontsize=10)
plt.tight_layout()

# 7. Save figure at 300 DPI
plt.savefig('figures/fig8_customer_spend_vs_visits.png', dpi=300, bbox_inches='tight')
print("Figure 8 saved cleanly to figures/fig8_customer_spend_vs_visits.png")



# =========================================================================
# PHASE 4 & 5: MODELLING AND BUSINESS QUESTIONS (03_models)
# =========================================================================

# Import needed libraries

# Load the cleaned dataset
df = pd.read_csv('cleaned_master.csv')
df['week_start'] = pd.to_datetime(df['week_start'])

print(f"Loaded cleaned_master.csv: {df.shape[0]:,} rows across {df['category'].nunique()} categories.")

# -------------------------------------------------------------------------
# Model A: Log-Log Price Elasticity by Product Category
# -------------------------------------------------------------------------

results = []

# Estimate log-log regression per category
for cat in df['category'].unique():
    cat_data = df[df['category'] == cat]
    
    # Fit OLS model: log_units ~ log_price_ratio
    model = smf.ols('log_units ~ log_price_ratio', data=cat_data).fit()
    
    # Extract coefficient, 95% CI, p-value, and R-squared
    coef = model.params['log_price_ratio']
    ci = model.conf_int().loc['log_price_ratio']
    
    results.append({
        'Category': cat,
        'Elasticity (Beta)': round(coef, 4),
        '95% CI Lower': round(ci[0], 4),
        '95% CI Upper': round(ci[1], 4),
        'p-value': model.pvalues['log_price_ratio'],
        'R-squared': round(model.rsquared, 4)
    })

# Convert to DataFrame and rank from most price-sensitive (most negative) to least
elasticity_table = pd.DataFrame(results).sort_values('Elasticity (Beta)').reset_index(drop=True)
elasticity_table.index = elasticity_table.index + 1  # 1-based ranking

print("=== Category Price Elasticity Ranking (Model A) ===")
print(elasticity_table[['Category', 'Elasticity (Beta)', '95% CI Lower', '95% CI Upper', 'R-squared']])

# -------------------------------------------------------------------------
# Step 3: Define Predictors & Compute Variance Inflation Factors (VIF)
# -------------------------------------------------------------------------

# 1. Construct a chronological weekly time trend (1 to 104)
df['week_start'] = pd.to_datetime(df['week_start'])
df['trend'] = (df['week_start'] - df['week_start'].min()).dt.days // 7 + 1

# 2. Binary conversion for indicator variables
df['has_parking_binary'] = (df['has_parking'] == 'Yes').astype(int)

# 3. Candidate predictor list including both operational scale metrics
candidate_features = [
    'log_price_ratio', 'promo_flag', 'size_sqm', 'staff_count',
    'foot_traffic_index', 'competitors_2km', 'has_parking_binary',
    'avg_temp_c', 'rainfall_mm', 'holiday_week', 'trend', 
    'sin_season', 'cos_season'
]

# 4. Compute VIF for all candidate variables
X_candidates = df[candidate_features].copy()
X_candidates = sm.add_constant(X_candidates)

vif_data = pd.DataFrame({
    'Variable': X_candidates.columns,
    'VIF': [variance_inflation_factor(X_candidates.values, i) for i in range(X_candidates.shape[1])]
})

# Display VIF excluding the intercept constant, sorted descending
vif_table = vif_data[vif_data['Variable'] != 'const'].sort_values('VIF', ascending=False).reset_index(drop=True)

print("=== Variance Inflation Factor (VIF) Audit ===")
print(vif_table.to_string(index=False))

# -------------------------------------------------------------------------
# Step 4: Compare Model B Specifications (With vs. Without staff_count)
# -------------------------------------------------------------------------

# Model 1: Full specification containing collinear pair (size_sqm and staff_count)
formula_collinear = (
    'log_units ~ log_price_ratio + promo_flag + size_sqm + staff_count + '
    'foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + '
    'rainfall_mm + holiday_week + trend + sin_season + cos_season'
)
model_with_staff = smf.ols(formula_collinear, data=df).fit()

# Model 2: Parsimonious specification retaining physical floor capacity (size_sqm)
formula_clean = (
    'log_units ~ log_price_ratio + promo_flag + size_sqm + '
    'foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + '
    'rainfall_mm + holiday_week + trend + sin_season + cos_season'
)
model_clean = smf.ols(formula_clean, data=df).fit()

# Comparison summary
comparison_df = pd.DataFrame({
    'Metric': ['R-squared', 'Adj. R-squared', 'AIC', 'BIC', 'size_sqm Coef', 'size_sqm Std Err'],
    'With staff_count (VIF ~ 33)': [
        round(model_with_staff.rsquared, 4),
        round(model_with_staff.rsquared_adj, 4),
        round(model_with_staff.aic, 1),
        round(model_with_staff.bic, 1),
        f"{model_with_staff.params['size_sqm']:.6f}",
        f"{model_with_staff.bse['size_sqm']:.7f}"
    ],
    'Without staff_count (VIF = 1.23)': [
        round(model_clean.rsquared, 4),
        round(model_clean.rsquared_adj, 4),
        round(model_clean.aic, 1),
        round(model_clean.bic, 1),
        f"{model_clean.params['size_sqm']:.6f}",
        f"{model_clean.bse['size_sqm']:.7f}"
    ]
})

print("=== Model Comparison: Multicollinearity Resolution ===")
print(comparison_df.to_string(index=False))

# -------------------------------------------------------------------------
# Step 5: Residual Diagnostics & Heteroscedasticity Assessment
# -------------------------------------------------------------------------

# 1. Extract fitted values and residuals from the clean Model B
fitted_vals = model_clean.fittedvalues
residuals = model_clean.resid

# 2. Plot Residuals vs. Fitted and Normal Q-Q Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Residuals vs Fitted Values
sns.scatterplot(
    x=fitted_vals,
    y=residuals,
    alpha=0.15,
    s=15,
    color='#1f77b4',
    ax=axes[0]
)
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title('Residuals vs. Fitted Values', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Fitted Values (log_units)')
axes[0].set_ylabel('Residuals')

# Panel 2: Normal Q-Q Plot
sm.qqplot(residuals, line='45', fit=True, ax=axes[1], markerfacecolor='#1f77b4', markeredgecolor='none', alpha=0.15, markersize=3)
axes[1].set_title('Normal Q-Q Plot', fontsize=11, fontweight='bold')
axes[1].get_lines()[1].set_color('red')
axes[1].get_lines()[1].set_linewidth(1.5)

plt.suptitle('Model B: Residual Diagnostic Plots', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save diagnostic figure at 300 DPI
plt.savefig('figures/model_b_diagnostics.png', dpi=300, bbox_inches='tight')
# 3. Breusch-Pagan Test for Heteroscedasticity
bp_stat, bp_pval, f_stat, f_pval = het_breuschpagan(residuals, model_clean.model.exog)

print("=== Breusch-Pagan Test Results ===")
print(f"Lagrange Multiplier Statistic: {bp_stat:.2f}")
print(f"p-value (LM Test):             {bp_pval:.4e}")
print(f"F-Statistic:                   {f_stat:.2f}")
print(f"p-value (F-Test):              {f_pval:.4e}")

if bp_pval < 0.05:
    print("\nResult: Significant heteroscedasticity detected (p < 0.05). Applying HC3 robust standard errors.")
else:
    print("\nResult: Homoscedasticity holds (p >= 0.05).")

# 4. Refit Model B with HC3 Robust Standard Errors
model_b_hc3 = smf.ols(formula_clean, data=df).fit(cov_type='HC3')

# 5. Side-by-side standard error comparison table
se_comparison = pd.DataFrame({
    'Parameter': model_clean.params.index,
    'Coefficient': model_clean.params.values.round(5),
    'OLS Std Err': model_clean.bse.values.round(6),
    'HC3 Std Err': model_b_hc3.bse.values.round(6),
    'OLS p-value': model_clean.pvalues.values.round(5),
    'HC3 p-value': model_b_hc3.pvalues.values.round(5)
})

print("\n=== Standard Errors Comparison (OLS vs. HC3 Robust) ===")
print(se_comparison.to_string(index=False))

# -------------------------------------------------------------------------
# Step 6: Testing Domain Interaction Terms
# -------------------------------------------------------------------------

# 1. Create binary category indicators for domain interactions
df['is_beverages'] = (df['category'] == 'Beverages').astype(int)
df['is_fresh_produce'] = (df['category'] == 'Fresh Produce').astype(int)

# 2. Base Model B specification (without interactions)
formula_base = (
    'log_units ~ log_price_ratio + promo_flag + size_sqm + '
    'foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + '
    'rainfall_mm + holiday_week + trend + sin_season + cos_season'
)
model_base = smf.ols(formula_base, data=df).fit()

# Interaction 1: Promotional Responsiveness across Categories (promo × category)
formula_int1 = (
    'log_units ~ log_price_ratio + promo_flag * C(category) + size_sqm + '
    'foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + '
    'rainfall_mm + holiday_week + trend + sin_season + cos_season'
)
model_int1 = smf.ols(formula_int1, data=df).fit()

# Interaction 2: Weather Sensitivity on Beverages (temperature × Beverages)
formula_int2 = (
    'log_units ~ log_price_ratio + promo_flag + size_sqm + '
    'foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c * is_beverages + '
    'rainfall_mm + holiday_week + trend + sin_season + cos_season'
)
model_int2 = smf.ols(formula_int2, data=df).fit()

# Interaction 3: Weather Sensitivity on Produce (rainfall × Fresh Produce)
formula_int3 = (
    'log_units ~ log_price_ratio + promo_flag + size_sqm + '
    'foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + '
    'rainfall_mm * is_fresh_produce + holiday_week + trend + sin_season + cos_season'
)
model_int3 = smf.ols(formula_int3, data=df).fit()

# Combined Specification: Use ':' for domain interactions to avoid duplicate category dummies
formula_all_int = (
    'log_units ~ log_price_ratio + promo_flag * C(category) + size_sqm + '
    'foot_traffic_index + competitors_2km + C(has_parking) + '
    'avg_temp_c + avg_temp_c:is_beverages + '
    'rainfall_mm + rainfall_mm:is_fresh_produce + '
    'holiday_week + trend + sin_season + cos_season'
)
model_all_int = smf.ols(formula_all_int, data=df).fit()

# 3. Model Comparison Table
interaction_comparison = pd.DataFrame({
    'Model Specification': [
        'Base Model B (No Interactions)',
        'Model B + (promo × category)',
        'Model B + (temperature × Beverages)',
        'Model B + (rainfall × Fresh Produce)',
        'Model B + All Three Interactions'
    ],
    'R-squared': [
        model_base.rsquared,
        model_int1.rsquared,
        model_int2.rsquared,
        model_int3.rsquared,
        model_all_int.rsquared
    ],
    'Adj. R-squared': [
        model_base.rsquared_adj,
        model_int1.rsquared_adj,
        model_int2.rsquared_adj,
        model_int3.rsquared_adj,
        model_all_int.rsquared_adj
    ],
    'AIC': [
        model_base.aic,
        model_int1.aic,
        model_int2.aic,
        model_int3.aic,
        model_all_int.aic
    ],
    'BIC': [
        model_base.bic,
        model_int1.bic,
        model_int2.bic,
        model_int3.bic,
        model_all_int.bic
    ]
})

print("=== Model Comparison: Testing Interaction Terms ===")
print(interaction_comparison.round(4).to_string(index=False))

# 4. Display specific interaction coefficients and p-values
print("\n--- Key Interaction Coefficients ---")
print(f"Temperature × Beverages:    Coef = {model_int2.params['avg_temp_c:is_beverages']:+.4f} (p = {model_int2.pvalues['avg_temp_c:is_beverages']:.2e})")
print(f"Rainfall × Fresh Produce:   Coef = {model_int3.params['rainfall_mm:is_fresh_produce']:+.4f} (p = {model_int3.pvalues['rainfall_mm:is_fresh_produce']:.2e})")

# -------------------------------------------------------------------------
# Step 7: Time-Based Validation Split & Out-of-Sample Benchmarking
# -------------------------------------------------------------------------

# 1. Temporal Split: Train on Weeks 1 to 80, Test on Weeks 81 to 104
train_df = df[df['trend'] <= 80].copy()
test_df = df[df['trend'] > 80].copy()

print(f"Training observations (Weeks  1-80):  {len(train_df):,} rows")
print(f"Testing observations  (Weeks 81-104): {len(test_df):,} rows")

# 2. Fit Model B specifications strictly on the training set
# Specification A: Clean Multivariate Baseline (No Interactions)
formula_base = (
    'log_units ~ log_price_ratio + promo_flag + size_sqm + '
    'foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + '
    'rainfall_mm + holiday_week + trend + sin_season + cos_season'
)
model_b_trained = smf.ols(formula_base, data=train_df).fit()

# Specification B: Full Interaction Specification
formula_all_int = (
    'log_units ~ log_price_ratio + promo_flag * C(category) + size_sqm + '
    'foot_traffic_index + competitors_2km + C(has_parking) + '
    'avg_temp_c + avg_temp_c:is_beverages + '
    'rainfall_mm + rainfall_mm:is_fresh_produce + '
    'holiday_week + trend + sin_season + cos_season'
)
model_all_trained = smf.ols(formula_all_int, data=train_df).fit()

# 3. Generate Predictions on the Test Set (Weeks 81-104)
# Naive Lag-4 Baseline prediction: log of units sold 4 weeks prior
y_test_log = test_df['log_units']
pred_baseline_log = test_df['units_lag_4'].apply(math.log)

pred_model_b_log = model_b_trained.predict(test_df)
pred_model_all_log = model_all_trained.predict(test_df)

# 4. Pure Pandas metric calculation function
def evaluate_forecast(y_actual, y_predicted):
    errors = y_actual - y_predicted
    mae = errors.abs().mean()
    rmse = math.sqrt((errors ** 2).mean())
    ss_res = (errors ** 2).sum()
    ss_tot = ((y_actual - y_actual.mean()) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot)
    return round(rmse, 4), round(mae, 4), round(r2, 4)

# 5. Compute performance metrics on the log_units scale
b_rmse, b_mae, b_r2 = evaluate_forecast(y_test_log, pred_baseline_log)
mb_rmse, mb_mae, mb_r2 = evaluate_forecast(y_test_log, pred_model_b_log)
all_rmse, all_mae, all_r2 = evaluate_forecast(y_test_log, pred_model_all_log)

# 6. Summary Comparison Table
test_evaluation_df = pd.DataFrame({
    'Model': [
        'Naive Baseline (Lag-4)',
        'Model B (Multivariate Demand)',
        'Model B (With All Interactions)'
    ],
    'Test RMSE': [b_rmse, mb_rmse, all_rmse],
    'Test MAE': [b_mae, mb_mae, all_mae],
    'Test Out-of-Sample R²': [b_r2, mb_r2, all_r2]
})

print("\n=== Out-of-Sample Model Performance on Test Set (Weeks 81-104) ===")
print(test_evaluation_df.to_string(index=False))

# -------------------------------------------------------------------------
# Question 1: Promotional Profitability & Margin Dilution Audit
# -------------------------------------------------------------------------

# 1. Aggregate key metrics across category and promotion status
promo_analysis = df.groupby(['category', 'promo_flag']).agg(
    total_units=('units_sold', 'sum'),
    total_revenue=('revenue_ngn', 'sum'),
    total_gross_profit=('gross_profit', 'sum'),
    avg_discount=('discount_pct', 'mean'),
    weekly_records=('units_sold', 'count')
).reset_index()

# 2. Derive weekly averages and margin rates
promo_analysis['mean_weekly_units'] = promo_analysis['total_units'] / promo_analysis['weekly_records']
promo_analysis['gross_margin_pct'] = (promo_analysis['total_gross_profit'] / promo_analysis['total_revenue']) * 100
promo_analysis['profit_per_unit_ngn'] = promo_analysis['total_gross_profit'] / promo_analysis['total_units']

# 3. Pivot to compute exact baseline vs promo lift side-by-side
pivot_units = promo_analysis.pivot(index='category', columns='promo_flag', values='mean_weekly_units')
pivot_margin = promo_analysis.pivot(index='category', columns='promo_flag', values='gross_margin_pct')
pivot_unit_profit = promo_analysis.pivot(index='category', columns='promo_flag', values='profit_per_unit_ngn')
pivot_total_profit = promo_analysis.pivot(index='category', columns='promo_flag', values='total_gross_profit')

# 4. Construct comprehensive executive summary table
promo_summary = pd.DataFrame({
    'Base Units/Wk': pivot_units[0].round(1),
    'Promo Units/Wk': pivot_units[1].round(1),
    'Volume Lift (%)': (((pivot_units[1] - pivot_units[0]) / pivot_units[0]) * 100).round(2),
    'Base Margin (%)': pivot_margin[0].round(2),
    'Promo Margin (%)': pivot_margin[1].round(2),
    'Margin Compression (pts)': (pivot_margin[1] - pivot_margin[0]).round(2),
    'Base Profit/Unit (NGN)': pivot_unit_profit[0].round(2),
    'Promo Profit/Unit (NGN)': pivot_unit_profit[1].round(2),
    'Total Promo Profit (M NGN)': (pivot_total_profit[1] / 1e6).round(2),
    'Total Base Profit (M NGN)': (pivot_total_profit[0] / 1e6).round(2)
}).sort_values('Volume Lift (%)', ascending=False)

# 5. Classify commercial promotion effectiveness
# Profitable lift requires: Volume Lift % > Breakeven Lift %
promo_summary['Effectiveness'] = promo_summary['Volume Lift (%)'].apply(
    lambda x: 'High Profit Driver' if x > 60 else ('Marginal / Selective' if x > 45 else 'Margin Dilutive')
)

print("=== Executive Table: Promotional Lift & Margin Compression ===")
print(promo_summary.to_string())

# -------------------------------------------------------------------------
# SKU-Level Audit: Identifying Value-Destroying Promotions
# -------------------------------------------------------------------------

sku_promo = df.groupby(['product_id', 'category', 'promo_flag']).agg(
    mean_units=('units_sold', 'mean'),
    mean_margin=('margin_pct', 'mean'),
    total_profit=('gross_profit', 'sum')
).unstack('promo_flag')

sku_eval = pd.DataFrame({
    'Category': sku_promo['mean_units'].index.map(df.drop_duplicates('product_id').set_index('product_id')['category']),
    'Observed Lift (%)': ((sku_promo['mean_units'][1] - sku_promo['mean_units'][0]) / sku_promo['mean_units'][0]) * 100,
    'Base Margin (%)': sku_promo['mean_margin'][0],
    'Promo Margin (%)': sku_promo['mean_margin'][1],
    'Total Profit Promo (NGN)': sku_promo['total_profit'][1],
    'Total Profit Base (NGN)': sku_promo['total_profit'][0]
})

# Required lift to preserve gross profit cash value given margin loss
sku_eval['Breakeven Lift (%)'] = (
    (sku_eval['Base Margin (%)'] - sku_eval['Promo Margin (%)']) / (sku_eval['Promo Margin (%)'] / 100)
)

# Flag SKUs that fail to meet breakeven lift
sku_eval['Promo Verdict'] = sku_eval.apply(
    lambda row: 'Value Accretive' if row['Observed Lift (%)'] >= row['Breakeven Lift (%)'] else 'Profit Destroyer',
    axis=1
)

sku_eval = sku_eval.sort_values('Observed Lift (%)', ascending=False)

print("\n=== Product-Level Promotional Verdict ===")
print(sku_eval[['Category', 'Observed Lift (%)', 'Breakeven Lift (%)', 'Promo Verdict']].round(2))

# -------------------------------------------------------------------------
# Question 2: Competitor Density, Foot Traffic & Cannibalisation
# -------------------------------------------------------------------------

# 1. Aggregate lifetime operational performance by store
store_summary = df.groupby([
    'store_id', 'city', 'area', 'store_type', 
    'size_sqm', 'staff_count', 'competitors_2km', 'has_parking'
]).agg(
    total_revenue_ngn=('revenue_ngn', 'sum'),
    total_units_sold=('units_sold', 'sum'),
    mean_foot_traffic=('foot_traffic_index', 'mean'),
    weekly_gross_profit=('gross_profit', 'mean'),
    total_weeks=('week_start', 'nunique')
).reset_index()

# 2. Derive unit economics and operational intensity ratios
store_summary['annualised_revenue_m_ngn'] = (store_summary['total_revenue_ngn'] / store_summary['total_weeks'] * 52) / 1e6
store_summary['rev_per_sqm'] = store_summary['total_revenue_ngn'] / store_summary['size_sqm']
store_summary['rev_per_staff_m_ngn'] = (store_summary['total_revenue_ngn'] / store_summary['staff_count']) / 1e6
store_summary['units_per_traffic_pt'] = store_summary['total_units_sold'] / (store_summary['mean_foot_traffic'] * store_summary['total_weeks'])

# 3. Bin competitor density into analytical competition brackets
store_summary['competitor_tier'] = pd.cut(
    store_summary['competitors_2km'],
    bins=[-0.1, 2, 5, 12],
    labels=['Low (0-2)', 'Moderate (3-5)', 'High (6+)']
)

# 4. Group by competitor bracket to assess direct economic impact
comp_tier_impact = store_summary.groupby('competitor_tier', observed=True).agg(
    store_count=('store_id', 'count'),
    mean_traffic=('mean_foot_traffic', 'mean'),
    mean_rev_per_sqm=('rev_per_sqm', 'mean'),
    mean_rev_per_staff_m=('rev_per_staff_m_ngn', 'mean'),
    mean_annual_revenue_m=('annualised_revenue_m_ngn', 'mean')
).round(2)

print("=== Competitor Density Impact Across Brackets ===")
print(comp_tier_impact.to_string())

# 5. Measure correlation between competitors_2km and key KPIs
kpis = ['mean_foot_traffic', 'rev_per_sqm', 'rev_per_staff_m_ngn', 'annualised_revenue_m_ngn']
corr_with_comp = store_summary[['competitors_2km'] + kpis].corr()['competitors_2km'].drop('competitors_2km').round(3)

print("\n=== Linear Correlation: Competitors within 2km vs. Store KPIs ===")
for kpi, r_val in corr_with_comp.items():
    print(f"Competitors vs {kpi:<24}: r = {r_val:+.3f}")

# -------------------------------------------------------------------------
# Visualization: Store Performance under Local Competition
# -------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Foot Traffic Index vs. Competitor Density by Store Type
sns.regplot(
    data=store_summary,
    x='competitors_2km',
    y='mean_foot_traffic',
    scatter=False,
    color='#e74c3c',
    ax=axes[0]
)
sns.scatterplot(
    data=store_summary,
    x='competitors_2km',
    y='mean_foot_traffic',
    hue='store_type',
    style='store_type',
    palette='deep',
    s=85,
    ax=axes[0]
)
axes[0].set_title('Foot Traffic Index vs. Competitors within 2km', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Competitor Outlets within 2km')
axes[0].set_ylabel('Mean Weekly Foot Traffic Index')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Panel 2: Revenue per Sqm vs. Competitor Density
sns.regplot(
    data=store_summary,
    x='competitors_2km',
    y='rev_per_sqm',
    scatter=False,
    color='#2ecc71',
    ax=axes[1]
)
sns.scatterplot(
    data=store_summary,
    x='competitors_2km',
    y='rev_per_sqm',
    hue='store_type',
    style='store_type',
    palette='deep',
    s=85,
    legend=False,
    ax=axes[1]
)
axes[1].set_title('Revenue per Sqm vs. Competitors within 2km', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Competitor Outlets within 2km')
axes[1].set_ylabel('Lifetime Revenue / Size Sqm (NGN)')
axes[1].yaxis.set_major_formatter('{x:,.0f}')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Figure Q2: Competitive Density & Spatial Performance', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save at 300 DPI
plt.savefig('figures/fig_q2_competition_cannibalisation.png', dpi=300, bbox_inches='tight')
print("Saved competitive analysis figure to figures/fig_q2_competition_cannibalisation.png")

# -------------------------------------------------------------------------
# Format Vulnerability: Revenue per Sqm Sensitivity to Competitors
# -------------------------------------------------------------------------

format_comp = store_summary.groupby(['store_type', 'competitor_tier'], observed=True).agg(
    store_count=('store_id', 'count'),
    mean_rev_per_sqm=('rev_per_sqm', 'mean'),
    mean_foot_traffic=('mean_foot_traffic', 'mean')
).round(1).reset_index()

print("=== Store Format Performance Across Competition Tiers ===")
print(format_comp.to_string(index=False))

# -------------------------------------------------------------------------
# Question 3: Perishability, Shelf Life & Seasonal Spoilage Risk
# -------------------------------------------------------------------------

# 1. Isolate perishable product categories (Fresh Produce and Dairy)
perishables = df[df['is_perishable'] == 1].copy()

# 2. Summary of baseline shelf-life parameters by category and SKU
sku_spoilage_profile = perishables.groupby(['category', 'product_id']).agg(
    shelf_life_days=('shelf_life_days', 'first'),
    unit_cost_ngn=('unit_cost_ngn', 'first'),
    list_price_ngn=('list_price_ngn', 'first'),
    mean_weekly_units=('units_sold', 'mean'),
    std_weekly_units=('units_sold', 'std')
).reset_index()

# 3. Calculate coefficient of variation (demand volatility)
sku_spoilage_profile['demand_cv'] = (
    sku_spoilage_profile['std_weekly_units'] / sku_spoilage_profile['mean_weekly_units']
).round(3)

# Effective replenishment cycle (Weekly store delivery cycle = 7 days)
sku_spoilage_profile['shelf_life_to_cycle_ratio'] = (
    sku_spoilage_profile['shelf_life_days'] / 7.0
).round(2)

print("=== Perishable SKU Spoilage Risk Profile ===")
print(sku_spoilage_profile.to_string(index=False))

# -------------------------------------------------------------------------
# Demand Volatility Under Seasonal Weather Swings
# -------------------------------------------------------------------------

# 1. Define weather shock thresholds using Pandas quantiles
temp_q75 = df['avg_temp_c'].quantile(0.75)
rain_q75 = df['rainfall_mm'].quantile(0.75)

# 2. Tag weather conditions
df['weather_regime'] = 'Baseline'
df['weather_regime'] = df['weather_regime'].mask(
    df['avg_temp_c'] > temp_q75, 'High Heat'
).mask(
    df['rainfall_mm'] > rain_q75, 'Heavy Rain'
).mask(
    (df['avg_temp_c'] > temp_q75) & (df['rainfall_mm'] > rain_q75), 'Heat + Rain'
)

# 3. Aggregate Fresh Produce and Dairy sales performance across regimes
weather_perish_impact = df[df['category'].isin(['Fresh Produce', 'Dairy'])].groupby(
    ['category', 'weather_regime']
).agg(
    mean_units=('units_sold', 'mean'),
    median_units=('units_sold', 'median'),
    unit_volatility=('units_sold', 'std'),
    mean_revenue=('revenue_ngn', 'mean'),
    observations=('units_sold', 'count')
).reset_index()

# 4. Compute deviation percentage from baseline
baseline_units = weather_perish_impact[weather_perish_impact['weather_regime'] == 'Baseline'].set_index('category')['mean_units']
weather_perish_impact['pct_deviation_from_base'] = weather_perish_impact.apply(
    lambda row: ((row['mean_units'] - baseline_units[row['category']]) / baseline_units[row['category']]) * 100,
    axis=1
).round(2)

print("\n=== Perishable Performance Across Weather Regimes ===")
print(weather_perish_impact[['category', 'weather_regime', 'mean_units', 'unit_volatility', 'pct_deviation_from_base']].to_string(index=False))

# -------------------------------------------------------------------------
# Estimated Spoilage Cost & Markdown Exposure Model
# -------------------------------------------------------------------------

# Assume standard retail buffer stock target: 1.5x standard deviation of weekly demand
spoilage_model = perishables.groupby(['category', 'product_id', 'store_type']).agg(
    mean_weekly_demand=('units_sold', 'mean'),
    demand_sd=('units_sold', 'std'),
    unit_cost=('unit_cost_ngn', 'first'),
    list_price=('list_price_ngn', 'first'),
    shelf_life=('shelf_life_days', 'first')
).reset_index()

# Buffer stock policy: 1.65 * SD (approx. 95% service level)
spoilage_model['buffer_units'] = (1.65 * spoilage_model['demand_sd']).round(1)

# In adverse weather (Heavy Rain), demand drops ~8-12%. 
# Units stranded on shelf beyond replenishment cycle are subject to waste:
spoilage_model['estimated_spoilage_pct'] = spoilage_model['shelf_life'].apply(
    lambda days: 0.12 if days <= 7 else (0.07 if days <= 14 else 0.03)
)

spoilage_model['weekly_spoiled_units'] = (spoilage_model['mean_weekly_demand'] * spoilage_model['estimated_spoilage_pct']).round(1)
spoilage_model['weekly_waste_cost_ngn'] = spoilage_model['weekly_spoiled_units'] * spoilage_model['unit_cost']
spoilage_model['annual_waste_cost_m_ngn'] = (spoilage_model['weekly_waste_cost_ngn'] * 52) / 1e6

# Summary by category
category_waste_summary = spoilage_model.groupby('category').agg(
    total_skus=('product_id', 'nunique'),
    mean_shelf_life=('shelf_life', 'mean'),
    weekly_waste_units=('weekly_spoiled_units', 'sum'),
    annual_waste_cost_m_ngn=('annual_waste_cost_m_ngn', 'sum')
).round(2).reset_index()

print("\n=== Annual Spoilage Cost Exposure (Chain-Level) ===")
print(category_waste_summary.to_string(index=False))

# -------------------------------------------------------------------------
# Figure Q3: Seasonal Perishability Volatility & Shelf-Life Stress (Fixed)
# -------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Panel 1: Demand Variation Across Weather Regimes for Perishables
sns.barplot(
    data=weather_perish_impact,
    x='weather_regime',
    y='mean_units',
    hue='category',
    palette=['#2ecc71', '#3498db'],
    ax=axes[0]
)
axes[0].set_title('Mean Weekly Sales Under Weather Regimes', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Weather Regime')
axes[0].set_ylabel('Mean Weekly Units Sold')
axes[0].legend(title='Category', loc='lower right')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Panel 2: Demand Volatility (Std Dev) vs. Shelf Life
sns.scatterplot(
    data=sku_spoilage_profile,
    x='shelf_life_days',
    y='std_weekly_units',
    hue='category',
    size='unit_cost_ngn',
    sizes=(60, 220),
    palette=['#2ecc71', '#3498db'],
    ax=axes[1]
)
axes[1].axvline(7, color='red', linestyle='--', linewidth=1.2, label='1-Week Delivery Cycle')
axes[1].set_title('Demand Volatility vs. Shelf-Life Thresholds', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Product Shelf Life (Days)')
axes[1].set_ylabel('Weekly Demand Standard Deviation (Units)')
axes[1].grid(True, linestyle='--', alpha=0.5)

# Place the legend cleanly outside the right border without axis clipping
sns.move_legend(
    axes[1],
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    title='Category & Cost'
)

plt.suptitle('Figure Q3: Perishability Exposure & Seasonal Weather Sensitivity', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save at 300 DPI with tight bounding box to accommodate the exterior legend
plt.savefig('figures/fig_q3_perishability_spoilage.png', dpi=300, bbox_inches='tight')
print("Saved clean Figure Q3 without legend overlap to figures/fig_q3_perishability_spoilage.png")

# -------------------------------------------------------------------------
# Question 4: Customer Segmentation & RFM Churn Analysis
# -------------------------------------------------------------------------

# 1. Load the Customer dataset
excel_file = 'Brainpals_sales_analytics.xlsx'
customers = pd.read_excel(excel_file, sheet_name='Customers')

# Identify relevant columns dynamically
spend_col = [c for c in customers.columns if 'spend' in c][0]
visits_col = [c for c in customers.columns if 'visit' in c][0]
tier_col = [c for c in customers.columns if 'tier' in c or 'loyalty' in c][0]
recency_col = [c for c in customers.columns if 'recency' in c or 'last' in c or 'days' in c]

# Standardise loyalty tier strings
customers[tier_col] = customers[tier_col].astype(str).str.strip().str.title()

# If an explicit recency column exists, use it; otherwise, derive days since last transaction
if recency_col:
    r_feature = recency_col[0]
else:
    # Use average inter-purchase interval proxy if recency is not directly recorded
    customers['days_since_last_visit'] = ((104 * 7) / (customers[visits_col] + 1)).round(1)
    r_feature = 'days_since_last_visit'

# 2. Compute RFM Quantiles (1 = Lowest, 4 = Highest)
# Recency: Lower days is better -> reverse scoring (qcut ranked inversely)
customers['R_Score'] = pd.qcut(customers[r_feature], q=4, labels=[4, 3, 2, 1]).astype(int)
customers['F_Score'] = pd.qcut(customers[visits_col].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
customers['M_Score'] = pd.qcut(customers[spend_col].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)

# Combined RFM Score string
customers['RFM_Segment'] = customers['R_Score'].astype(str) + customers['F_Score'].astype(str) + customers['M_Score'].astype(str)

# 3. Define Customer Behavioral Segments
def assign_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    if r >= 3 and f >= 3 and m >= 3:
        return 'Champions / Core Loyal'
    elif r >= 3 and f <= 2:
        return 'Recent / Promising'
    elif r <= 2 and (f >= 3 or m >= 3):
        return 'At Risk (High-Value Inactive)'
    elif r <= 2 and f <= 2 and m <= 2:
        return 'Hibernating / Lapsed'
    else:
        return 'Standard Active'

customers['Customer_Segment'] = customers.apply(assign_segment, axis=1)

# 4. Cohort Summary Table
segment_summary = customers.groupby('Customer_Segment').agg(
    customer_count=('customer_id' if 'customer_id' in customers.columns else visits_col, 'count'),
    mean_spend_ngn=(spend_col, 'mean'),
    total_spend_m_ngn=(spend_col, lambda x: x.sum() / 1e6),
    mean_visits=(visits_col, 'mean'),
    mean_recency_days=(r_feature, 'mean')
).round(2).sort_values('total_spend_m_ngn', ascending=False).reset_index()

# Calculate revenue share
segment_summary['revenue_share_pct'] = (
    (segment_summary['total_spend_m_ngn'] / segment_summary['total_spend_m_ngn'].sum()) * 100
).round(2)

print("=== Customer Behavioral Segmentation Summary ===")
print(segment_summary.to_string(index=False))

# -------------------------------------------------------------------------
# High-Value Churn Exposure Analysis
# -------------------------------------------------------------------------

# Cross-tabulate Loyalty Tier against Customer Behavioral Segment
tier_churn_crosstab = pd.crosstab(
    customers[tier_col],
    customers['Customer_Segment'],
    margins=True,
    margins_name='Total'
)

print("\n=== Customer Distribution: Loyalty Tier vs. Segment ===")
print(tier_churn_crosstab)

# Financial exposure from "At Risk" shoppers (Gold and Platinum)
at_risk_loyal = customers[
    (customers['Customer_Segment'] == 'At Risk (High-Value Inactive)') &
    (customers[tier_col].isin(['Gold', 'Platinum']))
]

print(f"\n--- At-Risk High-Value Shopper Audit ---")
print(f"Total High-Value At-Risk Customers: {len(at_risk_loyal):,}")
print(f"Total Cumulative Spend at Risk:     ₦{at_risk_loyal[spend_col].sum() / 1e6:,.2f} Million")
print(f"Mean Basket Value per Visit:        ₦{(at_risk_loyal[spend_col] / at_risk_loyal[visits_col]).mean():,.2f}")

# -------------------------------------------------------------------------
# Figure Q4: Customer Segmentation & Churn Risk Matrix (Clean)
# -------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Filter out margin totals if present
segment_plot_data = segment_summary[segment_summary['Customer_Segment'] != 'Total']

# Panel 1: Spend Share by Segment (single colour + tick_params avoids warnings)
sns.barplot(
    data=segment_plot_data,
    x='Customer_Segment',
    y='revenue_share_pct',
    color='#2b5c8f',
    ax=axes[0]
)
axes[0].set_title('Revenue Share (%) by Behavioral Cohort', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Behavioral Segment')
axes[0].set_ylabel('Total Chain Revenue Share (%)')
axes[0].tick_params(axis='x', rotation=20)
axes[0].grid(True, linestyle='--', alpha=0.5)

# Panel 2: Recency vs Spend with Churn Risk Boundary
sns.scatterplot(
    data=customers,
    x=r_feature,
    y=customers[spend_col] / 1e3,
    hue='Customer_Segment',
    palette='Set1',
    alpha=0.7,
    s=45,
    ax=axes[1]
)
axes[1].set_title('Customer Inactivity vs. Lifetime Spend', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Days Since Last Recorded Visit (Recency)')
axes[1].set_ylabel('Lifetime Spend (Thousand NGN)')
axes[1].grid(True, linestyle='--', alpha=0.5)

# Place legend cleanly outside without overlap
sns.move_legend(
    axes[1],
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    title='Behavioral Segment'
)

plt.suptitle('Figure Q4: Customer Retention Landscape & At-Risk Capital', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save at 300 DPI with tight bounding box
plt.savefig('figures/fig_q4_customer_churn_rfm.png', dpi=300, bbox_inches='tight')
print("Saved clean Figure Q4 to figures/fig_q4_customer_churn_rfm.png")

# -------------------------------------------------------------------------
# Question 5: Macro Shocks, Fuel Inflation & Pass-Through Elasticity
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Step 1: Macro Aggregation & Cost Ratios
# -------------------------------------------------------------------------


# 1. Aggregate to weekly chain-level series
macro_weekly = df.groupby('week_start').agg(
    cpi=('cpi_index', 'mean'),
    fuel_price=('fuel_price_ngn_per_l', 'mean'),
    mean_shelf_price=('unit_price_ngn', 'mean'),
    mean_unit_cost=('unit_cost_ngn', 'mean'),
    total_chain_units=('units_sold', 'sum'),
    total_chain_revenue=('revenue_ngn', 'sum'),
    total_chain_profit=('gross_profit', 'sum')
).reset_index()

# 2. Derive margin and cost-to-price ratios
macro_weekly['chain_margin_pct'] = (macro_weekly['total_chain_profit'] / macro_weekly['total_chain_revenue']) * 100
macro_weekly['cost_to_price_ratio'] = (macro_weekly['mean_unit_cost'] / macro_weekly['mean_shelf_price']) * 100

print(f"Aggregated {len(macro_weekly)} trading weeks of macroeconomic indicators.")

# -------------------------------------------------------------------------
# Step 2: Shelf Price Pass-Through Elasticity
# -------------------------------------------------------------------------

# 1. Calculate log values using Python's standard math library
macro_weekly['log_shelf_price'] = macro_weekly['mean_shelf_price'].apply(math.log)
macro_weekly['log_fuel'] = macro_weekly['fuel_price'].apply(math.log)
macro_weekly['log_cpi'] = macro_weekly['cpi'].apply(math.log)

# 2. Fit OLS pass-through model
pass_through_model = smf.ols('log_shelf_price ~ log_fuel + log_cpi', data=macro_weekly).fit()

print("=== Macro Pass-Through Elasticity to Retail Shelf Prices ===")
print(pass_through_model.summary().tables[1])

# -------------------------------------------------------------------------
# Step 3: Category-Level Fuel & CPI Elasticities
# -------------------------------------------------------------------------

cat_macro_results = []

for cat in df['category'].unique():
    cat_sub = df[df['category'] == cat].copy()
    cat_sub['log_fuel_local'] = cat_sub['fuel_price_ngn_per_l'].apply(math.log)
    cat_sub['log_cpi_local'] = cat_sub['cpi_index'].apply(math.log)
    
    cat_model = smf.ols(
        'log_units ~ log_price_ratio + promo_flag + log_fuel_local + log_cpi_local',
        data=cat_sub
    ).fit()
    
    cat_macro_results.append({
        'Category': cat,
        'Fuel Price Elasticity': round(cat_model.params['log_fuel_local'], 4),
        'Fuel p-value': round(cat_model.pvalues['log_fuel_local'], 4),
        'CPI Elasticity': round(cat_model.params['log_cpi_local'], 4),
        'CPI p-value': round(cat_model.pvalues['log_cpi_local'], 4)
    })

cat_macro_table = pd.DataFrame(cat_macro_results).sort_values('Fuel Price Elasticity').reset_index(drop=True)

print("=== Category Macro Elasticities ===")
print(cat_macro_table.to_string(index=False))

# -------------------------------------------------------------------------
# Step 4: Macro Stress Simulation
# -------------------------------------------------------------------------

fuel_shock_pct = 0.25   # +25% fuel increase
cpi_shock_pct = 0.15    # +15% headline inflation

log_fuel_delta = math.log(1.0 + fuel_shock_pct)
log_cpi_delta = math.log(1.0 + cpi_shock_pct)

# Project percentage volume change
cat_macro_table['Projected Volume Impact (%)'] = cat_macro_table.apply(
    lambda r: (math.exp(r['Fuel Price Elasticity'] * log_fuel_delta + r['CPI Elasticity'] * log_cpi_delta) - 1.0) * 100,
    axis=1
).round(2)

# Classify risk level
cat_macro_table['Shock Vulnerability'] = cat_macro_table['Projected Volume Impact (%)'].apply(
    lambda x: 'High Contraction' if x < -8.0 else ('Moderate Drag' if x < -4.0 else 'Defensive / Resilient')
)

print("=== Stress-Test Projections (+25% Fuel, +15% CPI) ===")
print(cat_macro_table[['Category', 'Fuel Price Elasticity', 'CPI Elasticity', 'Projected Volume Impact (%)', 'Shock Vulnerability']].to_string(index=False))

# -------------------------------------------------------------------------
# Step 5: Figure Q5 Visualization
# -------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left Panel: Cost-to-Price Ratio vs. Fuel Price
sns.regplot(
    data=macro_weekly,
    x='fuel_price',
    y='cost_to_price_ratio',
    scatter_kws={'alpha': 0.6, 'color': '#2b5c8f', 's': 35},
    line_kws={'color': '#d9534f', 'linewidth': 2},
    ax=axes[0]
)
axes[0].set_title('Fuel Pump Price vs. Unit Cost-to-Shelf-Price Ratio', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Fuel Price (NGN per Litre)')
axes[0].set_ylabel('Cost-to-Price Ratio (%) [Cost / Shelf Price]')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Right Panel: Projected Volume Contraction
plot_macro = cat_macro_table.sort_values('Projected Volume Impact (%)', ascending=True)

sns.barplot(
    data=plot_macro,
    x='Category',
    y='Projected Volume Impact (%)',
    color='#34495e',
    ax=axes[1]
)
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_title('Projected Unit Sales Contraction (+25% Fuel, +15% CPI)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Product Category')
axes[1].set_ylabel('Simulated Demand Volume Change (%)')
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Figure Q5: Macroeconomic Inflation & Fuel Shock Exposure', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save at 300 DPI
plt.savefig('figures/fig_q5_macro_fuel_shocks.png', dpi=300, bbox_inches='tight')
print("Figure Q5 exported cleanly to figures/fig_q5_macro_fuel_shocks.png")


# =============================================================================
# VERIFICATION AUDIT: ARTIFACT GENERATION
# =============================================================================
print("\n" + "=" * 70)
print("FINAL PIPELINE AUDIT")
print("=" * 70)

expected_figures = [
    'fig1_units_distribution.png',
    'fig2_store_correlation_heatmap.png',
    'fig3_weekly_revenue_trend.png',
    'fig4_units_by_category.png',
    'fig5_promo_lift_by_category.png',
    'fig6_sales_per_sqm_store_type.png',
    'fig7_log_log_demand_curve.png',
    'fig8_customer_spend_vs_visits.png',
    'fig_q2_competition_cannibalisation.png',
    'fig_q3_perishability_spoilage.png',
    'fig_q4_customer_churn_rfm.png',
    'fig_q5_macro_fuel_shocks.png'
]

missing_figs = [f for f in expected_figures if not os.path.exists(os.path.join('figures', f))]

if not missing_figs and os.path.exists('cleaned_master.csv'):
    print("SUCCESS: Pipeline executed end-to-end with zero errors!")
    print(f" - Cleaned master table: cleaned_master.csv ({master_df.shape[0]:,} rows)")
    print(f" - All {len(expected_figures)} high-resolution (300 DPI) figures verified in figures/")
else:
    print(f"WARNING: Pipeline finished with missing artifacts: {missing_figs}")

print("=" * 70 + "\n")