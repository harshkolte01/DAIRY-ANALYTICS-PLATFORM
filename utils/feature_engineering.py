"""
Feature Engineering Module for Dairy Analytics Platform
Advanced feature extraction and engineering from M5 dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Creates advanced features for ML models and business insights
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.is_fitted = False
    
    def create_time_features(self, df, date_col='date'):
        """Create comprehensive time-based features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Convert to datetime if needed
        if df[date_col].dtype == 'object':
            date_series = pd.to_datetime(df[date_col])
        else:
            date_series = df[date_col]
        
        # Basic time features
        features['year'] = date_series.dt.year
        features['month'] = date_series.dt.month
        features['day'] = date_series.dt.day
        features['day_of_week'] = date_series.dt.dayofweek
        features['day_of_year'] = date_series.dt.dayofyear
        features['week_of_year'] = date_series.dt.isocalendar().week
        features['quarter'] = date_series.dt.quarter
        
        # Advanced time features
        features['is_weekend'] = date_series.dt.dayofweek.isin([5, 6]).astype(int)
        features['is_month_start'] = date_series.dt.is_month_start.astype(int)
        features['is_month_end'] = date_series.dt.is_month_end.astype(int)
        features['is_quarter_start'] = date_series.dt.is_quarter_start.astype(int)
        features['is_quarter_end'] = date_series.dt.is_quarter_end.astype(int)
        features['is_year_start'] = date_series.dt.is_year_start.astype(int)
        features['is_year_end'] = date_series.dt.is_year_end.astype(int)
        
        # Days since/until key dates
        features['days_since_year_start'] = (date_series - date_series.dt.to_period('Y').dt.start_time).dt.days
        features['days_until_year_end'] = (date_series.dt.to_period('Y').dt.end_time - date_series).dt.days
        features['days_since_month_start'] = (date_series - date_series.dt.to_period('M').dt.start_time).dt.days
        features['days_until_month_end'] = (date_series.dt.to_period('M').dt.end_time - date_series).dt.days
        
        # Cyclical encoding for periodical features
        features['sin_day_of_week'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['cos_day_of_week'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['sin_month'] = np.sin(2 * np.pi * features['month'] / 12)
        features['cos_month'] = np.cos(2 * np.pi * features['month'] / 12)
        features['sin_day_of_year'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
        features['cos_day_of_year'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
        features['sin_week_of_year'] = np.sin(2 * np.pi * features['week_of_year'] / 52)
        features['cos_week_of_year'] = np.cos(2 * np.pi * features['week_of_year'] / 52)
        
        return features
    
    def create_lag_features(self, df, target_col, lags=[1, 2, 3, 7, 14, 21, 28, 35, 42]):
        """Create lag features for time series"""
        
        features = pd.DataFrame(index=df.index)
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"Warning: Target column '{target_col}' not found. Skipping lag features.")
            return features
        
        for lag in lags:
            features[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return features
    
    def create_rolling_features(self, df, target_col, windows=[3, 7, 14, 21, 28, 30, 60, 90]):
        """Create rolling statistical features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"Warning: Target column '{target_col}' not found. Skipping rolling features.")
            return features
        
        for window in windows:
            # Basic statistics
            features[f'{target_col}_{window}d_mean'] = df[target_col].rolling(window, min_periods=1).mean()
            features[f'{target_col}_{window}d_std'] = df[target_col].rolling(window, min_periods=1).std()
            features[f'{target_col}_{window}d_min'] = df[target_col].rolling(window, min_periods=1).min()
            features[f'{target_col}_{window}d_max'] = df[target_col].rolling(window, min_periods=1).max()
            features[f'{target_col}_{window}d_median'] = df[target_col].rolling(window, min_periods=1).median()
            
            # Advanced statistics (with proper min_periods validation)
            min_periods_skew = min(3, window)
            min_periods_kurt = min(4, window)
            features[f'{target_col}_{window}d_skew'] = df[target_col].rolling(window, min_periods=min_periods_skew).skew()
            features[f'{target_col}_{window}d_kurt'] = df[target_col].rolling(window, min_periods=min_periods_kurt).kurt()
            features[f'{target_col}_{window}d_q25'] = df[target_col].rolling(window, min_periods=1).quantile(0.25)
            features[f'{target_col}_{window}d_q75'] = df[target_col].rolling(window, min_periods=1).quantile(0.75)
            
            # Trend features
            features[f'{target_col}_{window}d_trend'] = df[target_col].rolling(window, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
            )
        
        return features.fillna(0)
    
    def create_price_features(self, df, price_col='sell_price'):
        """Create price-related features"""
        
        if price_col not in df.columns:
            return pd.DataFrame(index=df.index)
        
        features = pd.DataFrame(index=df.index)
        
        # Basic price features
        features['price'] = df[price_col]
        features['price_lag_1'] = df[price_col].shift(1)
        features['price_lag_7'] = df[price_col].shift(7)
        features['price_lag_14'] = df[price_col].shift(14)
        
        # Price changes
        features['price_change_1d'] = df[price_col].diff()
        features['price_change_7d'] = df[price_col] - df[price_col].shift(7)
        features['price_change_14d'] = df[price_col] - df[price_col].shift(14)
        features['price_change_pct_1d'] = df[price_col].pct_change()
        features['price_change_pct_7d'] = df[price_col].pct_change(7)
        
        # Price rolling statistics
        for window in [7, 14, 28]:
            features[f'price_{window}d_mean'] = df[price_col].rolling(window, min_periods=1).mean()
            features[f'price_{window}d_std'] = df[price_col].rolling(window, min_periods=1).std()
            features[f'price_relative_to_{window}d'] = df[price_col] / features[f'price_{window}d_mean']
        
        # Price volatility
        features['price_volatility_7d'] = features['price_change_pct_1d'].rolling(7).std()
        features['price_volatility_14d'] = features['price_change_pct_1d'].rolling(14).std()
        
        return features.fillna(method='bfill').fillna(0)
    
    def create_event_features(self, df):
        """Create event-related features"""
        
        features = pd.DataFrame(index=df.index)
        
        # SNAP features
        snap_cols = [col for col in df.columns if 'snap' in col.lower()]
        for col in snap_cols:
            if col in df.columns:
                features[col] = df[col].fillna(0)
        
        if snap_cols:
            features['total_snaps'] = df[snap_cols].fillna(0).sum(axis=1)
            features['snap_intensity'] = features['total_snaps'] / len(snap_cols)
        
        # Event type features
        event_cols = [col for col in df.columns if 'event_type' in col.lower()]
        for col in event_cols:
            if col in df.columns:
                features[f'has_{col}'] = (~df[col].isna()).astype(int)
        
        # Event name features (if available)
        event_name_cols = [col for col in df.columns if 'event_name' in col.lower()]
        for col in event_name_cols:
            if col in df.columns:
                features[f'has_{col}'] = (~df[col].isna()).astype(int)
        
        return features
    
    def create_statistical_features(self, df, target_col, groupby_cols=['item_id', 'store_id']):
        """Create statistical features based on historical patterns"""
        
        features = pd.DataFrame(index=df.index)
        
        for col in groupby_cols:
            if col in df.columns:
                # Historical statistics by group
                group_stats = df.groupby(col)[target_col].agg([
                    'mean', 'std', 'min', 'max', 'median', 'skew'
                ]).add_prefix(f'{col}_hist_')
                
                # Merge back to main dataframe
                features = features.join(df[col].map(group_stats.to_dict('index')), rsuffix=f'_{col}')
        
        # Z-score features (how unusual is current value)
        if f'{groupby_cols[0]}_hist_mean' in features.columns and f'{groupby_cols[0]}_hist_std' in features.columns:
            features[f'{target_col}_zscore'] = (
                df[target_col] - features[f'{groupby_cols[0]}_hist_mean']
            ) / (features[f'{groupby_cols[0]}_hist_std'] + 1e-8)
        
        return features.fillna(0)
    
    def create_interaction_features(self, df, feature_cols=None):
        """Create interaction features between important variables"""
        
        if feature_cols is None:
            # Default important feature combinations
            feature_cols = [col for col in df.columns if any(x in col.lower() for x in 
                          ['price', 'snap', 'event', 'day_of_week', 'month', 'weekend'])]
        
        features = pd.DataFrame(index=df.index)
        
        # Create pairwise interactions for selected features
        for i, col1 in enumerate(feature_cols[:10]):  # Limit to avoid explosion
            for col2 in feature_cols[i+1:10]:
                if col1 in df.columns and col2 in df.columns:
                    # Multiplicative interaction
                    features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return features
    
    def create_demand_pattern_features(self, df, target_col='sales'):
        """Create features that capture demand patterns"""
        
        features = pd.DataFrame(index=df.index)
        
        if target_col not in df.columns:
            return features
        
        # Demand momentum features
        features['demand_momentum_3d'] = df[target_col].diff(3)
        features['demand_momentum_7d'] = df[target_col].diff(7)
        features['demand_acceleration'] = features['demand_momentum_3d'].diff()
        
        # Demand consistency features
        features['demand_consistency_7d'] = 1 / (1 + df[target_col].rolling(7).std())
        features['demand_consistency_14d'] = 1 / (1 + df[target_col].rolling(14).std())
        
        # Peak detection features
        rolling_max_7d = df[target_col].rolling(7, center=True).max()
        rolling_max_14d = df[target_col].rolling(14, center=True).max()
        features['is_local_peak_7d'] = (df[target_col] == rolling_max_7d).astype(int)
        features['is_local_peak_14d'] = (df[target_col] == rolling_max_14d).astype(int)
        
        # Demand relative to historical performance
        features['demand_vs_hist_mean'] = df[target_col] / (df[target_col].expanding().mean() + 1e-8)
        features['demand_vs_recent_mean'] = df[target_col] / (df[target_col].rolling(30).mean() + 1e-8)
        
        return features.fillna(0)
    
    def fit_transform(self, df, target_col='sales', scale_features=True):
        """Create all features and optionally scale them"""
        
        all_features = pd.DataFrame(index=df.index)
        
        def safe_concat(existing_df, new_features):
            """Safely concatenate features avoiding duplicate columns"""
            if new_features is not None and not new_features.empty:
                # Remove any columns that already exist
                existing_cols = set(existing_df.columns)
                new_cols = [col for col in new_features.columns if col not in existing_cols]
                if new_cols:
                    return pd.concat([existing_df, new_features[new_cols]], axis=1)
            return existing_df
        
        # 1. Time features
        time_features = self.create_time_features(df)
        all_features = safe_concat(all_features, time_features)
        
        # 2. Lag features
        lag_features = self.create_lag_features(df, target_col)
        all_features = safe_concat(all_features, lag_features)
        
        # 3. Rolling features
        rolling_features = self.create_rolling_features(df, target_col)
        all_features = safe_concat(all_features, rolling_features)
        
        # 4. Price features
        price_features = self.create_price_features(df)
        all_features = safe_concat(all_features, price_features)
        
        # 5. Event features
        event_features = self.create_event_features(df)
        all_features = safe_concat(all_features, event_features)
        
        # 6. Statistical features
        if 'item_id' in df.columns or 'store_id' in df.columns:
            groupby_cols = [col for col in ['item_id', 'store_id'] if col in df.columns]
            stat_features = self.create_statistical_features(df, target_col, groupby_cols)
            all_features = safe_concat(all_features, stat_features)
        
        # 7. Demand pattern features
        pattern_features = self.create_demand_pattern_features(df, target_col)
        all_features = safe_concat(all_features, pattern_features)
        
        # 8. Interaction features (limited to avoid too many features)
        important_cols = [col for col in all_features.columns if any(x in col.lower() for x in 
                         ['price', 'snap', 'weekend', 'month'])][:8]
        if important_cols:
            # Create a clean DataFrame for interaction features to avoid duplicate column names
            interaction_df = df.copy()
            
            # Add important features, ensuring no duplicate columns
            for col in important_cols:
                if col not in interaction_df.columns:
                    interaction_df[col] = all_features[col]
            
            # Remove any duplicate columns by keeping the last occurrence
            interaction_df = interaction_df.loc[:, ~interaction_df.columns.duplicated(keep='last')]
            
            interaction_features = self.create_interaction_features(
                interaction_df, 
                important_cols
            )
            all_features = safe_concat(all_features, interaction_features)
        
        # Fill remaining NaN values
        all_features = all_features.fillna(0)
        
        # Scale features if requested
        if scale_features:
            numeric_cols = all_features.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            all_features[numeric_cols] = scaler.fit_transform(all_features[numeric_cols])
            self.scalers['main_scaler'] = scaler
        
        self.is_fitted = True
        self.feature_names = all_features.columns.tolist()
        
        return all_features
    
    def transform(self, df, target_col='sales', scale_features=True):
        """Transform new data using fitted parameters"""
        
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted first")
        
        # Apply same transformations as in fit_transform
        # (This is a simplified version - in practice, you'd want to store
        # the exact parameters used during fitting)
        
        return self.fit_transform(df, target_col, scale_features)


class BusinessInsightExtractor:
    """
    Extracts actionable business insights from engineered features
    """
    
    def __init__(self):
        self.insights = {}
    
    def analyze_demand_drivers(self, df, target_col='sales'):
        """Identify key drivers of demand"""
        
        insights = {}
        
        # Check if target column exists, if not try to find one or skip analysis
        if target_col not in df.columns:
            # Try common target column names
            possible_targets = ['sales', 'y', 'demand', 'volume']
            target_col = None
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
            
            # If still no target column found, look for numeric columns that could be targets
            if target_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                # Filter out obviously non-target columns
                excluded_patterns = ['year', 'month', 'day', 'week', 'quarter', 'id', 'price', 'snap']
                potential_targets = [col for col in numeric_cols 
                                   if not any(pattern in col.lower() for pattern in excluded_patterns)]
                
                if len(potential_targets) > 0:
                    target_col = potential_targets[0]  # Use the first suitable numeric column
                else:
                    # No suitable target found, return empty insights
                    return {
                        'note': 'No suitable target column found for demand analysis',
                        'seasonal_patterns': {},
                        'price_sensitivity': {},
                        'event_impact': {}
                    }
        
        # Price sensitivity analysis (only if we have a valid target)
        if target_col in df.columns and any('price' in col for col in df.columns):
            price_cols = [col for col in df.columns if 'price' in col and 'change' in col]
            if price_cols:
                price_sensitivity = {}
                for col in price_cols:
                    try:
                        correlation = df[col].corr(df[target_col])
                        if not pd.isna(correlation):
                            price_sensitivity[col] = correlation
                    except Exception:
                        pass  # Skip if correlation calculation fails
                
                insights['price_sensitivity'] = price_sensitivity
        
        # Event impact analysis (only if we have a valid target)
        if target_col in df.columns:
            snap_cols = [col for col in df.columns if 'snap' in col.lower()]
            if snap_cols:
                event_impact = {}
                for col in snap_cols:
                    try:
                        avg_with_event = df[df[col] == 1][target_col].mean()
                        avg_without_event = df[df[col] == 0][target_col].mean()
                        if not pd.isna(avg_with_event) and not pd.isna(avg_without_event) and avg_without_event != 0:
                            impact_pct = ((avg_with_event - avg_without_event) / avg_without_event) * 100
                            event_impact[col] = {
                                'impact_percentage': impact_pct,
                                'with_event_avg': avg_with_event,
                                'without_event_avg': avg_without_event
                            }
                    except Exception:
                        pass  # Skip if calculation fails
                
                insights['event_impact'] = event_impact
        
        # Seasonal patterns (only if we have a valid target)
        if target_col in df.columns and 'month' in df.columns:
            try:
                monthly_demand = df.groupby('month')[target_col].mean()
                peak_months = monthly_demand.nlargest(3).index.tolist()
                low_months = monthly_demand.nsmallest(3).index.tolist()
                
                insights['seasonality'] = {
                    'peak_months': peak_months,
                    'low_months': low_months,
                    'seasonal_variance': monthly_demand.std(),
                    'monthly_averages': monthly_demand.to_dict()
                }
            except Exception:
                pass  # Skip if seasonal analysis fails
        
        # Day of week patterns (only if we have a valid target)
        if target_col in df.columns and 'day_of_week' in df.columns:
            try:
                dow_demand = df.groupby('day_of_week')[target_col].mean()
                insights['day_of_week_patterns'] = {
                    'averages': dow_demand.to_dict(),
                    'best_day': dow_demand.idxmax(),
                    'worst_day': dow_demand.idxmin(),
                    'weekend_vs_weekday': {
                        'weekend_avg': df[df['is_weekend'] == 1][target_col].mean() if 'is_weekend' in df.columns else None,
                        'weekday_avg': df[df['is_weekend'] == 0][target_col].mean() if 'is_weekend' in df.columns else None
                    }
                }
            except Exception:
                pass  # Skip if day-of-week analysis fails
        
        return insights
    
    def identify_anomalies(self, df, target_col='sales'):
        """Identify demand anomalies and unusual patterns"""
        
        # Check if target column exists, if not try to find one or skip analysis
        if target_col not in df.columns:
            # Try common target column names
            possible_targets = ['sales', 'y', 'demand', 'volume']
            target_col = None
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
            
            # If still no target column found, look for numeric columns that could be targets
            if target_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                # Filter out obviously non-target columns
                excluded_patterns = ['year', 'month', 'day', 'week', 'quarter', 'id', 'price', 'snap']
                potential_targets = [col for col in numeric_cols 
                                   if not any(pattern in col.lower() for pattern in excluded_patterns)]
                
                if len(potential_targets) > 0:
                    target_col = potential_targets[0]  # Use the first suitable numeric column
                else:
                    # No suitable target found, return empty anomalies
                    return {
                        'note': 'No suitable target column found for anomaly detection',
                        'statistical_outliers': {'count': 0, 'percentage': 0},
                        'sudden_changes': {'count': 0, 'percentage': 0}
                    }
        
        anomalies = {}
        
        # Statistical outliers (only if we have a valid target)
        if target_col in df.columns:
            try:
                Q1 = df[target_col].quantile(0.25)
                Q3 = df[target_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
                
                anomalies['statistical_outliers'] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'outlier_dates': outliers.index.tolist() if hasattr(outliers.index, 'tolist') else [],
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
            except Exception:
                anomalies['statistical_outliers'] = {'count': 0, 'percentage': 0}
        
        # Sudden changes (only if we have a valid target)
        if target_col in df.columns and len(df) > 1:
            try:
                daily_changes = df[target_col].diff().abs()
                large_changes = daily_changes > daily_changes.quantile(0.95)
                
                anomalies['sudden_changes'] = {
                    'count': large_changes.sum(),
                    'percentage': (large_changes.sum() / len(df)) * 100,
                    'threshold': daily_changes.quantile(0.95),
                    'change_dates': df[large_changes].index.tolist() if hasattr(df.index, 'tolist') else []
                }
            except Exception:
                anomalies['sudden_changes'] = {'count': 0, 'percentage': 0}
        
        return anomalies
    
    def generate_recommendations(self, insights, anomalies):
        """Generate actionable business recommendations"""
        
        recommendations = []
        
        # Price-based recommendations
        if 'price_sensitivity' in insights:
            for price_feature, sensitivity in insights['price_sensitivity'].items():
                if sensitivity < -0.3:  # Strong negative correlation
                    recommendations.append({
                        'type': 'pricing',
                        'priority': 'high',
                        'message': f"Strong price sensitivity detected ({sensitivity:.2f}). Consider dynamic pricing strategies.",
                        'action': 'Implement price optimization algorithms'
                    })
        
        # Event-based recommendations
        if 'event_impact' in insights:
            for event, impact_data in insights['event_impact'].items():
                if impact_data['impact_percentage'] > 20:
                    recommendations.append({
                        'type': 'inventory',
                        'priority': 'high',
                        'message': f"{event} increases demand by {impact_data['impact_percentage']:.1f}%",
                        'action': f"Increase inventory by 25-30% during {event} events"
                    })
        
        # Seasonal recommendations
        if 'seasonality' in insights:
            peak_months = insights['seasonality']['peak_months']
            recommendations.append({
                'type': 'capacity',
                'priority': 'medium',
                'message': f"Peak demand months: {peak_months}",
                'action': "Plan capacity expansion and maintenance during low-demand periods"
            })
        
        # Anomaly-based recommendations
        if anomalies.get('statistical_outliers', {}).get('percentage', 0) > 5:
            recommendations.append({
                'type': 'quality',
                'priority': 'medium',
                'message': f"{anomalies['statistical_outliers']['percentage']:.1f}% of data points are outliers",
                'action': "Investigate data quality and demand forecasting accuracy"
            })
        
        return recommendations


def create_comprehensive_features(sales_data, calendar_data, prices_data=None):
    """Create comprehensive feature set for ML models"""
    
    # Check if we have raw M5 data or processed time series data
    if 'd' in sales_data.columns:
        # Raw M5 data - merge on 'd' column
        merged_data = sales_data.merge(calendar_data, on='d', how='left')
        if prices_data is not None:
            merged_data = merged_data.merge(
                prices_data, 
                on=['item_id', 'store_id', 'wm_yr_wk'], 
                how='left'
            )
    else:
        # Processed time series data - create a different merge strategy
        merged_data = sales_data.copy()
        
        # Add date column if not present
        if 'date' not in merged_data.columns and 'ds' in merged_data.columns:
            merged_data['date'] = merged_data['ds']
        elif 'date' not in merged_data.columns:
            # Create date range starting from M5 dataset start
            merged_data['date'] = pd.date_range(
                start='2011-01-29', 
                periods=len(merged_data)
            )
        
        # Convert date to datetime if needed
        if merged_data['date'].dtype == 'object':
            merged_data['date'] = pd.to_datetime(merged_data['date'])
        
        # Merge with calendar data based on date
        calendar_subset = calendar_data.copy()
        if 'date' in calendar_subset.columns:
            calendar_subset['date'] = pd.to_datetime(calendar_subset['date'])
            merged_data = merged_data.merge(
                calendar_subset, 
                on='date', 
                how='left'
            )
        
        # Add pricing data if available (simplified approach for time series)
        if prices_data is not None:
            # For processed time series, we'll add average pricing information
            avg_prices = prices_data.groupby(['item_id'])['sell_price'].agg(['mean', 'std']).reset_index()
            avg_prices.columns = ['item_id', 'avg_sell_price', 'price_volatility']
            
            # If we have item_id in the data, merge; otherwise add default values
            if 'item_id' in merged_data.columns:
                merged_data = merged_data.merge(avg_prices, on='item_id', how='left')
            else:
                # Add average prices for the main food item used in processing
                food_prices = avg_prices[avg_prices['item_id'] == 'FOODS_3_090']
                if not food_prices.empty:
                    merged_data['avg_sell_price'] = food_prices['avg_sell_price'].iloc[0]
                    merged_data['price_volatility'] = food_prices['price_volatility'].iloc[0]
                else:
                    merged_data['avg_sell_price'] = 1.0  # Default price
                    merged_data['price_volatility'] = 0.1  # Default volatility
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Detect target column name
    target_col = None
    if 'sales' in merged_data.columns:
        target_col = 'sales'
    elif 'y' in merged_data.columns:
        target_col = 'y'
    else:
        # Try to find a numeric column that could be the target
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
        potential_targets = [col for col in numeric_cols if col not in ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter']]
        if len(potential_targets) > 0:
            target_col = potential_targets[0]
        else:
            # Create a dummy target column for feature engineering
            merged_data['sales'] = 1.0  # Dummy values
            target_col = 'sales'
    
    # Create features
    features = feature_engineer.fit_transform(merged_data, target_col=target_col)
    
    # Extract insights
    insight_extractor = BusinessInsightExtractor()
    
    # Create combined data for insights, avoiding duplicate columns
    combined_data = merged_data.copy()
    feature_cols = [col for col in features.columns if col not in combined_data.columns]
    if feature_cols:
        combined_data = pd.concat([combined_data, features[feature_cols]], axis=1)
    
    demand_insights = insight_extractor.analyze_demand_drivers(combined_data)
    anomalies = insight_extractor.identify_anomalies(merged_data)
    recommendations = insight_extractor.generate_recommendations(demand_insights, anomalies)
    
    return {
        'features': features,
        'insights': demand_insights,
        'anomalies': anomalies,
        'recommendations': recommendations,
        'feature_engineer': feature_engineer
    }
