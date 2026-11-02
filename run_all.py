"""
Brainpals retail sales analytics: end-to-end pipeline (run_all.py)
Reproduces 01_cleaning.ipynb, 02_eda.ipynb and 03_models.ipynb from the raw Excel file.
Needs Brainpals_sales_analytics.xlsx in the same folder. Writes cleaned_master.csv and figures/.
"""

import math
import os
import warnings

import matplotlib
matplotlib.use('Agg')   # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')



# =========================================================================
# PHASES 1 AND 2: CLEANING AND FEATURE ENGINEERING
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
# PHASE 3: EXPLORATORY DATA ANALYSIS
# =========================================================================

sns.set_theme(style='whitegrid')
os.makedirs('figures', exist_ok=True)

df = pd.read_csv('cleaned_master.csv')
df['week_start'] = pd.to_datetime(df['week_start'])

print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")

weekly = df.groupby('week_start')['revenue_ngn'].sum().reset_index()
weekly['revenue_m'] = weekly['revenue_ngn'] / 1e6
weekly['rolling_12'] = weekly['revenue_m'].rolling(12, min_periods=1).mean()

parts = seasonal_decompose(weekly.set_index('week_start')['revenue_m'], model='additive', period=52)
weekly['trend'] = parts.trend.values
weekly['seasonal'] = parts.seasonal.values
weekly['residual'] = parts.resid.values

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

sns.lineplot(data=weekly, x='week_start', y='revenue_m', label='Weekly revenue', ax=axes[0])
sns.lineplot(data=weekly, x='week_start', y='rolling_12', label='12-week rolling mean', ax=axes[0])
axes[0].set(title='Figure 1: Chain-Wide Weekly Revenue and Decomposition', ylabel='Revenue (million NGN)')

sns.lineplot(data=weekly, x='week_start', y='trend', ax=axes[1], color='green')
axes[1].set(ylabel='Trend (million NGN)')

sns.lineplot(data=weekly, x='week_start', y='seasonal', ax=axes[2], color='purple')
axes[2].set(ylabel='Seasonality (million NGN)')

sns.scatterplot(data=weekly, x='week_start', y='residual', ax=axes[3], color='red')
axes[3].axhline(0, color='black', linestyle='--')
axes[3].set(ylabel='Residual (million NGN)', xlabel='Week starting (date)')

plt.tight_layout()
plt.savefig('figures/fig1_revenue_ts.png', dpi=300)

store_external_vars = [
    'size_sqm', 'staff_count', 'foot_traffic_index', 'competitors_2km', 'manager_tenure_yrs',
    'avg_temp_c', 'rainfall_mm', 'cpi_index', 'fuel_price_ngn_per_l', 'holiday_week'
]

plt.figure(figsize=(10, 8))
sns.heatmap(df[store_external_vars].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, cbar_kws={'label': 'Pearson correlation'})
plt.title('Figure 2: Correlation of Store and External Variables')
plt.tight_layout()
plt.savefig('figures/fig2_correlation_heatmap.png', dpi=300)

df['promo_label'] = df['promo_flag'].map({0: 'No promo', 1: 'Promo'})

g = sns.catplot(data=df, x='promo_label', y='units_sold', hue='promo_label', col='category',
                col_wrap=4, kind='box', showfliers=False, height=3.5, aspect=0.9, legend=False)
g.set_axis_labels('Promotion status', 'Units sold per store-week')
g.set_titles('{col_name}')
g.figure.suptitle('Figure 3: Weekly Units Sold by Category, Promo vs Non-Promo', y=1.03)
g.savefig('figures/fig3_promo_by_category.png', dpi=300, bbox_inches='tight')

stores = df.groupby(['store_id', 'store_type', 'size_sqm', 'staff_count'])['revenue_ngn'].sum().reset_index()
stores['revenue_per_sqm'] = stores['revenue_ngn'] / stores['size_sqm']
stores['revenue_per_staff_m'] = stores['revenue_ngn'] / stores['staff_count'] / 1e6
stores = stores.sort_values('revenue_per_sqm', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)

sns.barplot(data=stores, y='store_id', x='revenue_per_sqm', hue='store_type', dodge=False, ax=axes[0])
axes[0].set(title='Revenue per sqm', xlabel='Two-year revenue per sqm (NGN)', ylabel='Store')

sns.barplot(data=stores, y='store_id', x='revenue_per_staff_m', hue='store_type', dodge=False, ax=axes[1])
axes[1].set(title='Revenue per staff member', xlabel='Two-year revenue per staff member (million NGN)', ylabel='')
axes[1].get_legend().remove()

plt.suptitle('Figure 4: Store Productivity Ranking (ordered by revenue per sqm)')
plt.tight_layout()
plt.savefig('figures/fig4_store_productivity.png', dpi=300)

print('Rank correlation between the two measures:',
      round(stores['revenue_per_sqm'].corr(stores['revenue_per_staff_m'], method='spearman'), 2))

g = sns.lmplot(data=df, x='log_price_ratio', y='log_units', col='category', col_wrap=4,
               height=3.2, scatter_kws={'alpha': 0.1, 's': 8}, line_kws={'color': 'red'})
g.set_axis_labels('log price ratio (log of real price / list price)', 'log units sold')
g.set_titles('{col_name}')
g.figure.suptitle('Figure 5: Log Units vs Log Price Ratio by Category', y=1.03)
g.savefig('figures/fig5_log_price_vs_log_units.png', dpi=300, bbox_inches='tight')

order = ['Express', 'Standard', 'Hypermarket']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.boxplot(data=df, x='store_type', y='units_sold', order=order, showfliers=False, ax=axes[0])
axes[0].set(title='Boxplot (outliers hidden)', xlabel='Store type', ylabel='Units sold per store-week')

sns.violinplot(data=df, x='store_type', y='units_sold', order=order, cut=0, inner='quartile', ax=axes[1])
axes[1].set_yscale('log')
axes[1].set(title='Violin plot (log scale)', xlabel='Store type', ylabel='Units sold per store-week (log scale)')

plt.suptitle('Figure 6: Weekly Units Sold by Store Type')
plt.tight_layout()
plt.savefig('figures/fig6_units_by_store_type.png', dpi=300)

corr_vars = [
    'units_sold', 'unit_price_ngn', 'discount_pct', 'promo_flag', 'size_sqm', 'foot_traffic_index',
    'competitors_2km', 'avg_temp_c', 'rainfall_mm', 'cpi_index', 'fuel_price_ngn_per_l'
]

pearson = df[corr_vars].corr(method='pearson')
spearman = df[corr_vars].corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.heatmap(pearson, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0,
            cbar=False, square=True, ax=axes[0])
axes[0].set_title('Pearson (straight-line)')
sns.heatmap(spearman, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, cbar_kws={'label': 'Correlation'}, ax=axes[1])
axes[1].set_title('Spearman (rank-based)')
plt.suptitle('Figure 7: Pearson vs Spearman Correlation')
plt.tight_layout()
plt.savefig('figures/fig7_pearson_vs_spearman.png', dpi=300)

# Table of the pairs where the two methods disagree most
table = pd.DataFrame({'pearson': pearson.unstack(), 'spearman': spearman.unstack()}).reset_index()
table.columns = ['variable_1', 'variable_2', 'pearson', 'spearman']
table['gap'] = (table['pearson'] - table['spearman']).abs()
table = table[table['variable_1'] < table['variable_2']]
print((table.sort_values('gap', ascending=False).head(8).round(2).reset_index(drop=True)).to_string())

customers = pd.read_excel('Brainpals_sales_analytics.xlsx', sheet_name='Customers')
customers['spend_k_ngn'] = customers['total_spend_12m_ngn'] / 1e3

plt.figure(figsize=(10, 6))
sns.scatterplot(data=customers, x='visits_12m', y='spend_k_ngn', hue='loyalty_tier',
                hue_order=['Bronze', 'Silver', 'Gold'], palette='viridis', alpha=0.7)
plt.title('Figure 8: Customer Spend vs Visits by Loyalty Tier')
plt.xlabel('Store visits in last 12 months (count)')
plt.ylabel('Total spend in last 12 months (thousand NGN)')
plt.legend(title='Loyalty tier')
plt.tight_layout()
plt.savefig('figures/fig8_customer_spend_vs_visits.png', dpi=300)



# =========================================================================
# PHASES 4 AND 5: MODELS AND BUSINESS QUESTIONS
# =========================================================================

sns.set_theme(style='whitegrid')
os.makedirs('figures', exist_ok=True)

df = pd.read_csv('cleaned_master.csv')
df['week_start'] = pd.to_datetime(df['week_start'])

# Extra columns used in the models
df['trend'] = (df['week_start'] - df['week_start'].min()).dt.days // 7 + 1   # week number 1 to 104
df['has_parking_binary'] = (df['has_parking'] == 'Yes').astype(int)
df['is_beverages'] = (df['category'] == 'Beverages').astype(int)
df['is_fresh_produce'] = (df['category'] == 'Fresh Produce').astype(int)

print(f"Loaded {df.shape[0]:,} rows across {df['category'].nunique()} categories.")

rows = []
for category, data in df.groupby('category'):
    model = smf.ols('log_units ~ log_price_ratio', data=data).fit()
    low, high = model.conf_int().loc['log_price_ratio']
    rows.append({
        'category': category,
        'elasticity': round(model.params['log_price_ratio'], 3),
        'ci_95_lower': round(low, 3),
        'ci_95_upper': round(high, 3),
        'r_squared': round(model.rsquared, 3)
    })

elasticity = pd.DataFrame(rows).sort_values('elasticity').reset_index(drop=True)
elasticity.index = elasticity.index + 1   # rank 1 = most price-sensitive
print((elasticity).to_string())

predictors = ['log_price_ratio', 'promo_flag', 'size_sqm', 'staff_count', 'foot_traffic_index',
              'competitors_2km', 'has_parking_binary', 'avg_temp_c', 'rainfall_mm',
              'holiday_week', 'trend', 'sin_season', 'cos_season']

X = sm.add_constant(df[predictors])
vif = pd.DataFrame({
    'predictor': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
vif = vif[vif['predictor'] != 'const'].sort_values('VIF', ascending=False).round(2)
print(vif.to_string(index=False))
print()
print('Correlation between size_sqm and staff_count:', round(df['size_sqm'].corr(df['staff_count']), 3))

formula_with_staff = 'log_units ~ log_price_ratio + promo_flag + size_sqm + staff_count + foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + rainfall_mm + holiday_week + trend + sin_season + cos_season'
formula_clean = 'log_units ~ log_price_ratio + promo_flag + size_sqm + foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + rainfall_mm + holiday_week + trend + sin_season + cos_season'

model_with_staff = smf.ols(formula_with_staff, data=df).fit()
model_clean = smf.ols(formula_clean, data=df).fit()

print((pd.DataFrame({
    'with staff_count': [model_with_staff.rsquared, model_with_staff.aic, model_with_staff.params['size_sqm'], model_with_staff.bse['size_sqm']],
    'without staff_count': [model_clean.rsquared, model_clean.aic, model_clean.params['size_sqm'], model_clean.bse['size_sqm']]
}, index=['R-squared', 'AIC', 'size_sqm coefficient', 'size_sqm standard error']).round(7)).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(x=model_clean.fittedvalues, y=model_clean.resid, alpha=0.1, s=10, ax=axes[0])
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set(title='Residuals vs fitted', xlabel='Fitted value (log units)', ylabel='Residual (log units)')

sm.qqplot(model_clean.resid, line='45', fit=True, ax=axes[1], markersize=3, alpha=0.15)
axes[1].set(title='Normal Q-Q plot', xlabel='Theoretical quantiles (standard deviations)', ylabel='Sample quantiles (standard deviations)')

plt.suptitle('Model B: Residual Diagnostics')
plt.tight_layout()
plt.savefig('figures/model_b_diagnostics.png', dpi=300)

lm_stat, lm_p, f_stat, f_p = het_breuschpagan(model_clean.resid, model_clean.model.exog)
print(f"Breusch-Pagan: LM = {lm_stat:.2f}, p-value = {lm_p:.2e}")
if lm_p < 0.05:
    print('Heteroscedasticity found: using robust (HC3) standard errors.')
else:
    print('No heteroscedasticity found: ordinary standard errors are fine.')

model_robust = smf.ols(formula_clean, data=df).fit(cov_type='HC3')

print((pd.DataFrame({
    'coefficient': model_clean.params,
    'OLS std error': model_clean.bse,
    'HC3 std error': model_robust.bse,
    'HC3 p-value': model_robust.pvalues
}).round(5)).to_string())

base = smf.ols(formula_clean, data=df).fit()

interaction_formulas = {
    'promo x category': formula_clean + ' + C(category) + promo_flag:C(category)',
    'temperature x Beverages': formula_clean + ' + is_beverages + avg_temp_c:is_beverages',
    'rainfall x Fresh Produce': formula_clean + ' + is_fresh_produce + rainfall_mm:is_fresh_produce',
    'all three': formula_clean + ' + C(category) + promo_flag:C(category) + avg_temp_c:is_beverages + rainfall_mm:is_fresh_produce'
}

rows = [{'model': 'Base Model B', 'R-squared': base.rsquared, 'adj R-squared': base.rsquared_adj,
         'AIC': base.aic, 'BIC': base.bic, 'F-test p-value': None, 'improves model?': ''}]

for name, formula in interaction_formulas.items():
    model = smf.ols(formula, data=df).fit()
    p_value = sm.stats.anova_lm(base, model)['Pr(>F)'].iloc[1]
    rows.append({'model': name, 'R-squared': model.rsquared, 'adj R-squared': model.rsquared_adj,
                 'AIC': model.aic, 'BIC': model.bic, 'F-test p-value': p_value,
                 'improves model?': 'Yes' if p_value < 0.05 and model.aic < base.aic else 'No'})

print((pd.DataFrame(rows).round(4)).to_string())

train = df[df['trend'] <= 80]
test = df[df['trend'] > 80]
print(f"Train: {len(train):,} rows (weeks 1-80). Test: {len(test):,} rows (weeks 81-104).")

model_b_train = smf.ols(formula_clean, data=train).fit()
model_all_train = smf.ols(interaction_formulas['all three'], data=train).fit()

predictions = {
    'Naive lag-4 baseline': test['units_lag_4'].apply(math.log),
    'Model B': model_b_train.predict(test),
    'Model B + interactions': model_all_train.predict(test)
}

def score(actual, predicted):
    error = actual - predicted
    rmse = (error ** 2).mean() ** 0.5
    mae = error.abs().mean()
    r2 = 1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    return rmse, mae, r2

results = pd.DataFrame(
    [score(test['log_units'], predicted) for predicted in predictions.values()],
    index=predictions.keys(), columns=['RMSE', 'MAE', 'R-squared']
).round(4)
print((results).to_string())

plot_data = results.reset_index().rename(columns={'index': 'model'})

plt.figure(figsize=(8, 4))
sns.barplot(data=plot_data, x='model', y='RMSE', hue='model', legend=False)
plt.title('Test-Set Error by Model (weeks 81 to 104)')
plt.xlabel('Model')
plt.ylabel('RMSE (log units; lower is better)')
plt.tight_layout()
plt.savefig('figures/model_test_rmse.png', dpi=300)

def promo_table(by):
    means = df.groupby(by + ['promo_flag'])[['units_sold', 'gross_profit']].mean().unstack()
    out = pd.DataFrame({
        'units_base': means['units_sold'][0], 'units_promo': means['units_sold'][1],
        'profit_base': means['gross_profit'][0], 'profit_promo': means['gross_profit'][1]
    })
    out['volume_lift_pct'] = (out['units_promo'] / out['units_base'] - 1) * 100
    out['breakeven_lift_pct'] = ((out['profit_base'] / out['units_base']) / (out['profit_promo'] / out['units_promo']) - 1) * 100
    out['profit_change_pct'] = (out['profit_promo'] / out['profit_base'] - 1) * 100
    out['verdict'] = out['profit_change_pct'].apply(lambda x: 'Loses money' if x < 0 else 'Makes money')
    return out.round(1)

by_category = promo_table(['category']).sort_values('profit_change_pct', ascending=False)
by_product = promo_table(['category', 'product_id']).sort_values('profit_change_pct', ascending=False)

print(by_category[['volume_lift_pct', 'breakeven_lift_pct', 'profit_change_pct', 'verdict']].to_string())
print()
print(by_product[['volume_lift_pct', 'breakeven_lift_pct', 'profit_change_pct', 'verdict']].to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(data=by_category.reset_index(), y='category', x='profit_change_pct', color='steelblue', ax=axes[0])
axes[0].axvline(0, color='black')
axes[0].set(title='By category', xlabel='Change in gross profit during promotion (%)', ylabel='Category')

product_plot = by_product.reset_index()
product_plot['product'] = product_plot['product_id'] + ' (' + product_plot['category'] + ')'
sns.barplot(data=product_plot, y='product', x='profit_change_pct', color='steelblue', ax=axes[1])
axes[1].axvline(0, color='black')
axes[1].set(title='By product', xlabel='Change in gross profit during promotion (%)', ylabel='Product')

plt.suptitle('Question 1: Profit Effect of Promotions (per store-product-week)')
plt.tight_layout()
plt.savefig('figures/fig_q1_promo_profit.png', dpi=300)

discounts = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
controls = ('size_sqm + foot_traffic_index + competitors_2km + C(has_parking) + avg_temp_c + '
            'rainfall_mm + holiday_week + trend + sin_season + cos_season')

rows = []
betas = {}
for category, data in df.groupby('category'):
    model = smf.ols('log_units ~ discount_pct + ' + controls, data=data).fit()
    beta = model.params['discount_pct']
    betas[category] = round(beta, 2)
    cost_share = (data['unit_cost_ngn'] / data['list_price_ngn']).mean()
    for d in discounts:
        units_index = math.exp(beta * d)
        rows.append({
            'category': category,
            'discount_pct': d * 100,
            'revenue_index': (1 - d) * units_index,
            'profit_index': units_index * ((1 - d) - cost_share) / (1 - cost_share)
        })

curves = pd.DataFrame(rows)

best_revenue = curves.loc[curves.groupby('category')['revenue_index'].idxmax(), ['category', 'discount_pct', 'revenue_index']]
best_profit = curves.loc[curves.groupby('category')['profit_index'].idxmax(), ['category', 'discount_pct', 'profit_index']]

answer = best_revenue.merge(best_profit, on='category', suffixes=('_revenue', '_profit'))
answer['discount_beta'] = answer['category'].map(betas)
print((answer.round(2).rename(columns={'discount_pct_revenue': 'revenue-maximising discount (%)',
                                'discount_pct_profit': 'profit-maximising discount (%)'})).to_string())

print('Observations at each discount level:')
print((df['discount_pct'] * 100).round().astype(int).value_counts().sort_index().to_string())

plt.figure(figsize=(9, 5))
sns.lineplot(data=curves, x='discount_pct', y='revenue_index', hue='category', marker='o')
plt.title('Question 2: Revenue Index by Discount Level')
plt.xlabel('Discount (% off list price)')
plt.ylabel('Revenue index (1.00 = no discount)')
plt.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/fig_q2_revenue_vs_discount.png', dpi=300)

df['residual'] = model_clean.resid

store_gap = df.groupby('store_id').agg(
    mean_residual=('residual', 'mean'),
    std_residual=('residual', 'std'),
    rows=('residual', 'size'),
    city=('city', 'first'),
    store_type=('store_type', 'first'),
    size_sqm=('size_sqm', 'first'),
    foot_traffic=('foot_traffic_index', 'first'),
    competitors=('competitors_2km', 'first')
).reset_index()

store_gap['gap_vs_prediction_pct'] = store_gap['mean_residual'].apply(lambda r: (math.exp(r) - 1) * 100)
store_gap['t_statistic'] = store_gap['mean_residual'] / (store_gap['std_residual'] / store_gap['rows'] ** 0.5)
store_gap = store_gap.sort_values('gap_vs_prediction_pct').reset_index(drop=True)

print((store_gap.head(5)[['store_id', 'city', 'store_type', 'size_sqm', 'foot_traffic', 'competitors',
                   'gap_vs_prediction_pct', 't_statistic']].round(1)).to_string())

store_gap['group'] = store_gap.index.map(lambda i: 'Bottom 5' if i < 5 else 'Other stores')

plt.figure(figsize=(9, 9))
sns.barplot(data=store_gap, y='store_id', x='gap_vs_prediction_pct', hue='group',
            palette={'Bottom 5': 'crimson', 'Other stores': 'lightgray'}, dodge=False)
plt.axvline(0, color='black')
plt.title('Question 3: Store Sales vs Model B Prediction')
plt.xlabel('Actual sales vs prediction (%)')
plt.ylabel('Store')
plt.legend(title='')
plt.tight_layout()
plt.savefig('figures/fig_q3_store_residuals.png', dpi=300)

weekly = df.groupby('week_start').agg(
    revenue=('revenue_ngn', 'sum'), units=('units_sold', 'sum'), cpi=('cpi_index', 'mean')
).reset_index()
weekly['real_revenue'] = weekly['revenue'] / weekly['cpi'] * 100

first = weekly.head(13).mean(numeric_only=True)
last = weekly.tail(13).mean(numeric_only=True)

nominal = last['revenue'] / first['revenue'] - 1
volume = last['units'] / first['units'] - 1
price_per_unit = (last['revenue'] / last['units']) / (first['revenue'] / first['units']) - 1
inflation = last['cpi'] / first['cpi'] - 1
real = last['real_revenue'] / first['real_revenue'] - 1
price_mix = (1 + price_per_unit) / (1 + inflation) - 1

def share(growth):
    return math.log(1 + growth) / math.log(1 + nominal) * 100

summary = pd.DataFrame({
    'growth_pct': [nominal * 100, volume * 100, price_per_unit * 100, inflation * 100, price_mix * 100, real * 100],
    'share_of_nominal_growth_pct': [100, share(volume), share(price_per_unit), share(inflation), share(price_mix), None]
}, index=['Nominal revenue', 'Real volume (units sold)', 'Average price per unit', 'Inflation (CPI)',
          'Price and mix beyond inflation', 'Real revenue (CPI-adjusted)']).round(1)
print((summary).to_string())

indexed = weekly.copy()
for column in ['revenue', 'units', 'cpi', 'real_revenue']:
    indexed[column] = indexed[column] / first[column] * 100

indexed = indexed.rename(columns={'revenue': 'Nominal revenue', 'units': 'Units sold',
                                  'cpi': 'CPI (inflation)', 'real_revenue': 'Real revenue (CPI-adjusted)'})
indexed = indexed.melt(id_vars='week_start', value_vars=['Nominal revenue', 'Units sold', 'CPI (inflation)',
                                                         'Real revenue (CPI-adjusted)'],
                       var_name='measure', value_name='index')

plt.figure(figsize=(11, 5))
sns.lineplot(data=indexed, x='week_start', y='index', hue='measure')
plt.title('Question 4: Revenue Growth, Volume and Inflation')
plt.xlabel('Week starting (date)')
plt.ylabel('Index (average of first 13 weeks = 100)')
plt.legend(title='')
plt.tight_layout()
plt.savefig('figures/fig_q4_growth_decomposition.png', dpi=300)

customers = pd.read_excel('Brainpals_sales_analytics.xlsx', sheet_name='Customers')
customers['age'] = customers['age'].where(customers['age'].between(1, 100))   # impossible ages become blank

# RFM scores: 4 is best. Recency: fewer days since the last visit is better.
customers['R'] = pd.qcut(customers['days_since_last_visit'], q=4, labels=[4, 3, 2, 1]).astype(int)
customers['F'] = pd.qcut(customers['visits_12m'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
customers['M'] = pd.qcut(customers['total_spend_12m_ngn'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)

def segment(row):
    if row['R'] >= 3 and row['F'] >= 3 and row['M'] >= 3:
        return 'Champions'
    elif row['R'] >= 3 and row['F'] <= 2:
        return 'Recent / Promising'
    elif row['R'] <= 2 and (row['F'] >= 3 or row['M'] >= 3):
        return 'At Risk (high value, inactive)'
    elif row['R'] <= 2 and row['F'] <= 2 and row['M'] <= 2:
        return 'Hibernating'
    else:
        return 'Standard Active'

customers['segment'] = customers.apply(segment, axis=1)

segments = customers.groupby('segment').agg(
    customers=('customer_id', 'count'),
    churn_rate_pct=('churned_next_90d', lambda s: s.mean() * 100),
    total_spend_m_ngn=('total_spend_12m_ngn', lambda s: s.sum() / 1e6)
).sort_values('churn_rate_pct', ascending=False).round(1)

print(f"Overall churn rate: {customers['churned_next_90d'].mean() * 100:.1f}%")
print()
print(segments)
print()
print(customers.groupby('loyalty_tier')['churned_next_90d'].mean().mul(100).round(1).to_string())

# Logistic regression. Total spend is left out because it equals visits x basket size.
features = ['days_since_last_visit', 'visits_12m', 'avg_basket_ngn', 'discount_usage_ratio',
            'tenure_yrs', 'age', 'loyalty_member']

model_data = customers[features + ['churned_next_90d']].copy()
model_data['loyalty_member'] = (model_data['loyalty_member'] == 'Yes').astype(int)
model_data = model_data.dropna()

# Standardise every driver so the coefficients can be compared fairly
for column in features:
    model_data[column] = (model_data[column] - model_data[column].mean()) / model_data[column].std()

logit = smf.logit('churned_next_90d ~ ' + ' + '.join(features), data=model_data).fit(disp=0)

drivers = pd.DataFrame({
    'driver': features,
    'coefficient': logit.params[features].values,
    'odds_ratio': logit.params[features].apply(math.exp).values,
    'p_value': logit.pvalues[features].values
})
drivers['strength'] = drivers['coefficient'].abs()
drivers = drivers.sort_values('strength', ascending=False).drop(columns='strength').reset_index(drop=True)

print(f"Customers used: {len(model_data):,}. Pseudo R-squared: {logit.prsquared:.3f}")
print((drivers.round(3)).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=segments.reset_index(), y='segment', x='churn_rate_pct', color='steelblue', ax=axes[0])
axes[0].axvline(customers['churned_next_90d'].mean() * 100, color='red', linestyle='--', label='Overall average')
axes[0].set(title='Churn rate by segment', xlabel='Churned within 90 days (%)', ylabel='Segment')
axes[0].legend()

sns.barplot(data=drivers, y='driver', x='coefficient', color='steelblue', ax=axes[1])
axes[1].axvline(0, color='black')
axes[1].set(title='Drivers of churn (standardised)', xlabel='Effect on log-odds of churn (per 1 std. dev.)', ylabel='Driver')

plt.suptitle('Question 5: Customer Churn Risk')
plt.tight_layout()
plt.savefig('figures/fig_q5_churn_drivers.png', dpi=300)
