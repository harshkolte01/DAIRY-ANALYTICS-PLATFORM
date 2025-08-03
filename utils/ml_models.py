"""
ML Models for Dairy Analytics Platform
Advanced machine learning models trained on M5 dataset for enhanced business insights
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class DemandSpikeClassifier:
    """
    Classifies whether demand will spike, drop, or remain normal
    Uses features like day of week, month, events, seasonality
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        
    def prepare_features(self, sales_data, calendar_data):
        """Extract features for demand spike classification"""
        
        # Handle different data formats
        if 'd' in sales_data.columns:
            # Raw M5 data format
            merged_data = sales_data.merge(calendar_data, on='d', how='left')
        else:
            # Processed time series format
            merged_data = sales_data.copy()
            
            # Ensure we have a date column
            if 'date' not in merged_data.columns:
                if 'ds' in merged_data.columns:
                    merged_data['date'] = merged_data['ds']
                else:
                    # Create date range for M5 dataset
                    merged_data['date'] = pd.date_range(
                        start='2011-01-29', 
                        periods=len(merged_data)
                    )
            
            # Convert to datetime if needed
            if merged_data['date'].dtype == 'object':
                merged_data['date'] = pd.to_datetime(merged_data['date'])
            
            # Merge with calendar data on date
            calendar_subset = calendar_data.copy()
            if 'date' in calendar_subset.columns:
                calendar_subset['date'] = pd.to_datetime(calendar_subset['date'])
                merged_data = merged_data.merge(calendar_subset, on='date', how='left')
        
        features = pd.DataFrame()
        
        # Time-based features
        date_col = pd.to_datetime(merged_data['date']) if 'date' in merged_data.columns else pd.date_range(start='2011-01-29', periods=len(merged_data))
        features['day_of_week'] = date_col.dt.dayofweek
        features['month'] = date_col.dt.month
        features['quarter'] = date_col.dt.quarter
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Event features (with fallback values if not available)
        features['snap_CA'] = merged_data.get('snap_CA', 0).fillna(0)
        features['snap_TX'] = merged_data.get('snap_TX', 0).fillna(0)
        features['snap_WI'] = merged_data.get('snap_WI', 0).fillna(0)
        
        # Event type encoding (with fallback)
        event_encoder = LabelEncoder()
        features['event_type_1'] = event_encoder.fit_transform(
            merged_data.get('event_type_1', 'none').fillna('none')
        )
        features['event_type_2'] = event_encoder.fit_transform(
            merged_data.get('event_type_2', 'none').fillna('none')
        )
        event_encoder = LabelEncoder()
        features['event_type_1'] = event_encoder.fit_transform(
            merged_data['event_type_1'].fillna('none')
        )
        features['event_type_2'] = event_encoder.fit_transform(
            merged_data['event_type_2'].fillna('none')
        )
        
        # Rolling statistics (last 7 days)
        if 'sales' in merged_data.columns:
            features['sales_7d_mean'] = merged_data['sales'].rolling(7, min_periods=1).mean()
            features['sales_7d_std'] = merged_data['sales'].rolling(7, min_periods=1).std().fillna(0)
            features['sales_14d_mean'] = merged_data['sales'].rolling(14, min_periods=1).mean()
            features['sales_30d_mean'] = merged_data['sales'].rolling(30, min_periods=1).mean()
        
        # Price features (if available)
        if 'sell_price' in merged_data.columns:
            features['sell_price'] = merged_data['sell_price'].fillna(
                merged_data['sell_price'].mean()
            )
            features['price_7d_mean'] = features['sell_price'].rolling(7, min_periods=1).mean()
        
        # Seasonal features
        features['sin_day'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['cos_day'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['sin_month'] = np.sin(2 * np.pi * features['month'] / 12)
        features['cos_month'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features.fillna(0)
    
    def create_target_labels(self, sales_data, spike_threshold=1.5, drop_threshold=0.7):
        """Create target labels: 0=Normal, 1=Spike, 2=Drop"""
        
        # Handle different target column names
        if 'sales' in sales_data.columns:
            sales_col = sales_data['sales']
        elif 'y' in sales_data.columns:
            sales_col = sales_data['y']
        else:
            # Try to find a numeric column
            numeric_cols = sales_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sales_col = sales_data[numeric_cols[0]]
            else:
                raise ValueError("Could not find sales/target column")
        
        # Calculate rolling mean for comparison
        rolling_mean = sales_col.rolling(14, min_periods=7).mean()
        
        # Create spike/drop labels
        labels = []
        for i, (current_sales, avg_sales) in enumerate(zip(sales_col, rolling_mean)):
            if pd.isna(avg_sales):
                labels.append(0)  # Normal
            elif current_sales > avg_sales * spike_threshold:
                labels.append(1)  # Spike
            elif current_sales < avg_sales * drop_threshold:
                labels.append(2)  # Drop
            else:
                labels.append(0)  # Normal
        
        return np.array(labels)
    
    def train(self, sales_data, calendar_data):
        """Train the demand spike classifier"""
        
        # Prepare features and labels
        features = self.prepare_features(sales_data, calendar_data)
        labels = self.create_target_labels(sales_data)
        
        # Remove rows with insufficient data
        valid_idx = ~pd.isna(features).any(axis=1)
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, labels)
        self.feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        
        # Calculate accuracy
        predictions = self.model.predict(features_scaled)
        accuracy = (predictions == labels).mean()
        
        return {
            'accuracy': accuracy,
            'feature_importance': self.feature_importance,
            'class_distribution': pd.Series(labels).value_counts().to_dict()
        }
    
    def predict(self, sales_data, calendar_data):
        """Predict demand spikes/drops"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = self.prepare_features(sales_data, calendar_data)
        features_scaled = self.scaler.transform(features.fillna(0))
        
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'spike_probability': probabilities[:, 1] if probabilities.shape[1] > 1 else np.zeros(len(predictions)),
            'drop_probability': probabilities[:, 2] if probabilities.shape[1] > 2 else np.zeros(len(predictions))
        }


class DemandVolumeRegressor:
    """
    Predicts exact demand volumes using advanced regression techniques
    """
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def prepare_advanced_features(self, sales_data, calendar_data, prices_data=None):
        """Create advanced features for volume prediction"""
        
        # Handle different data formats
        if 'd' in sales_data.columns:
            # Raw M5 data format
            merged_data = sales_data.merge(calendar_data, on='d', how='left')
            if prices_data is not None:
                merged_data = merged_data.merge(prices_data, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')
        else:
            # Processed time series format
            merged_data = sales_data.copy()
            
            # Ensure we have a date column
            if 'date' not in merged_data.columns:
                if 'ds' in merged_data.columns:
                    merged_data['date'] = merged_data['ds']
                else:
                    merged_data['date'] = pd.date_range(
                        start='2011-01-29', 
                        periods=len(merged_data)
                    )
            
            # Convert to datetime if needed
            if merged_data['date'].dtype == 'object':
                merged_data['date'] = pd.to_datetime(merged_data['date'])
            
            # Merge with calendar data
            calendar_subset = calendar_data.copy()
            if 'date' in calendar_subset.columns:
                calendar_subset['date'] = pd.to_datetime(calendar_subset['date'])
                merged_data = merged_data.merge(calendar_subset, on='date', how='left')
            
            # Add simplified pricing if available
            if prices_data is not None:
                # Add average pricing for time series data
                avg_price = prices_data['sell_price'].mean()
                merged_data['sell_price'] = avg_price
        
        features = pd.DataFrame()
        
        # Basic time features (with fallback date creation)
        if 'date' in merged_data.columns:
            date_col = pd.to_datetime(merged_data['date'])
        else:
            date_col = pd.date_range(start='2011-01-29', periods=len(merged_data))
            
        features['day_of_week'] = date_col.dt.dayofweek
        features['month'] = date_col.dt.month
        features['quarter'] = date_col.dt.quarter
        try:
            # Try new pandas API
            features['week_of_year'] = date_col.dt.isocalendar().week
        except AttributeError:
            # Fallback for older pandas versions
            features['week_of_year'] = date_col.dt.week
        
        # Advanced time features
        features['is_month_start'] = date_col.dt.is_month_start.astype(int)
        features['is_month_end'] = date_col.dt.is_month_end.astype(int)
        features['is_quarter_start'] = date_col.dt.is_quarter_start.astype(int)
        features['is_quarter_end'] = date_col.dt.is_quarter_end.astype(int)
        
        # Event features (with fallbacks)
        features['snap_CA'] = merged_data.get('snap_CA', 0).fillna(0)
        features['snap_TX'] = merged_data.get('snap_TX', 0).fillna(0)
        features['snap_WI'] = merged_data.get('snap_WI', 0).fillna(0)
        features['total_snaps'] = features[['snap_CA', 'snap_TX', 'snap_WI']].sum(axis=1)
        
        # Event encoding (with fallbacks)
        features['has_event_1'] = (~merged_data.get('event_type_1', pd.Series([None]*len(merged_data))).isna()).astype(int)
        features['has_event_2'] = (~merged_data.get('event_type_2', pd.Series([None]*len(merged_data))).isna()).astype(int)
        
        # Rolling statistics with multiple windows
        if 'sales' in merged_data.columns:
            for window in [3, 7, 14, 21, 30]:
                features[f'sales_{window}d_mean'] = merged_data['sales'].rolling(window, min_periods=1).mean()
                features[f'sales_{window}d_std'] = merged_data['sales'].rolling(window, min_periods=1).std().fillna(0)
                features[f'sales_{window}d_min'] = merged_data['sales'].rolling(window, min_periods=1).min()
                features[f'sales_{window}d_max'] = merged_data['sales'].rolling(window, min_periods=1).max()
        
        # Lag features
        if 'sales' in merged_data.columns:
            for lag in [1, 2, 3, 7, 14, 21, 28]:
                features[f'sales_lag_{lag}'] = merged_data['sales'].shift(lag)
        
        # Price features
        if 'sell_price' in merged_data.columns:
            features['sell_price'] = merged_data['sell_price'].fillna(merged_data['sell_price'].mean())
            features['price_7d_mean'] = features['sell_price'].rolling(7, min_periods=1).mean()
            features['price_14d_mean'] = features['sell_price'].rolling(14, min_periods=1).mean()
            features['price_change_1d'] = features['sell_price'].diff()
            features['price_change_7d'] = features['sell_price'] - features['price_7d_mean']
        
        # Cyclical features
        features['sin_day'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['cos_day'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['sin_month'] = np.sin(2 * np.pi * features['month'] / 12)
        features['cos_month'] = np.cos(2 * np.pi * features['month'] / 12)
        features['sin_week'] = np.sin(2 * np.pi * features['week_of_year'] / 52)
        features['cos_week'] = np.cos(2 * np.pi * features['week_of_year'] / 52)
        
        return features.fillna(0)
    
    def train(self, sales_data, calendar_data, prices_data=None):
        """Train the volume regression model"""
        
        # Prepare features
        features = self.prepare_advanced_features(sales_data, calendar_data, prices_data)
        
        # Handle different target column names
        if 'sales' in sales_data.columns:
            target = sales_data['sales'].values
        elif 'y' in sales_data.columns:
            target = sales_data['y'].values
        else:
            # Try to find a numeric column that could be the target
            numeric_cols = sales_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target = sales_data[numeric_cols[0]].values
            else:
                raise ValueError("Could not find target column (sales, y, or any numeric column)")
        
        # Remove rows with insufficient data (first 30 days for lag features)
        valid_idx = 30
        features = features.iloc[valid_idx:]
        target = target[valid_idx:]
        
        # Scale features for non-tree models
        if self.model_type not in ['xgboost', 'lightgbm', 'random_forest']:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = features.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, target, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = None
        
        self.is_trained = True
        self.feature_names = features.columns.tolist()
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance
        }
    
    def predict(self, sales_data, calendar_data, prices_data=None):
        """Predict demand volumes"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = self.prepare_advanced_features(sales_data, calendar_data, prices_data)
        
        if self.model_type not in ['xgboost', 'lightgbm', 'random_forest']:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values
        
        predictions = self.model.predict(features_scaled)
        
        return {
            'predictions': predictions,
            'features': features
        }


class SeasonalityAnalyzer:
    """
    Analyzes complex seasonal patterns in demand data
    """
    
    def __init__(self):
        self.seasonal_components = None
        self.trend_components = None
        self.is_analyzed = False
    
    def analyze_seasonality(self, sales_data, calendar_data):
        """Perform advanced seasonality analysis"""
        
        # Handle different data formats
        if 'd' in sales_data.columns:
            # Raw M5 data format
            merged_data = sales_data.merge(calendar_data, on='d', how='left')
            merged_data['date'] = pd.to_datetime(merged_data['date'])
        else:
            # Processed time series format
            merged_data = sales_data.copy()
            
            # Ensure we have proper date and sales columns
            if 'ds' in merged_data.columns and 'sales' in merged_data.columns:
                merged_data['date'] = pd.to_datetime(merged_data['ds'])
            elif 'date' not in merged_data.columns:
                # Create date range
                merged_data['date'] = pd.date_range(
                    start='2011-01-29', 
                    periods=len(merged_data)
                )
        
        # Prepare target column
        if 'sales' in merged_data.columns:
            target_col = 'sales'
        elif 'y' in merged_data.columns:
            target_col = 'y'
        else:
            # Find first numeric column
            numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
            else:
                raise ValueError("Could not find sales/target column for seasonality analysis")
        
        # Prepare for Prophet analysis
        prophet_data = merged_data[['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})
        
        # Fit Prophet model for decomposition
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_data)
        
        # Get components
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        
        # Extract seasonality patterns
        self.seasonal_components = {
            'yearly': forecast[['ds', 'yearly']].copy(),
            'weekly': forecast[['ds', 'weekly']].copy(),
            'trend': forecast[['ds', 'trend']].copy()
        }
        
        # Calculate seasonal strength
        seasonal_strength = {
            'yearly_strength': forecast['yearly'].std(),
            'weekly_strength': forecast['weekly'].std(),
            'trend_strength': forecast['trend'].std()
        }
        
        # Identify peak seasons
        yearly_avg = forecast.groupby(forecast['ds'].dt.month)['yearly'].mean()
        peak_months = yearly_avg.nlargest(3).index.tolist()
        low_months = yearly_avg.nsmallest(3).index.tolist()
        
        weekly_avg = forecast.groupby(forecast['ds'].dt.dayofweek)['weekly'].mean()
        peak_days = weekly_avg.nlargest(3).index.tolist()
        low_days = weekly_avg.nsmallest(3).index.tolist()
        
        self.is_analyzed = True
        
        return {
            'seasonal_strength': seasonal_strength,
            'peak_months': peak_months,
            'low_months': low_months,
            'peak_days': peak_days,
            'low_days': low_days,
            'components': self.seasonal_components
        }
    
    def get_seasonal_multipliers(self, dates):
        """Get seasonal multipliers for given dates"""
        if not self.is_analyzed:
            raise ValueError("Seasonality analysis must be performed first")
        
        # Convert dates to datetime if needed
        if isinstance(dates, (list, np.ndarray)):
            dates = pd.to_datetime(dates)
        else:
            dates = pd.to_datetime([dates])
        
        multipliers = []
        for date in dates:
            # Find closest date in components
            yearly_mult = self.seasonal_components['yearly'].iloc[
                (self.seasonal_components['yearly']['ds'] - date).abs().argmin()
            ]['yearly']
            
            weekly_mult = self.seasonal_components['weekly'].iloc[
                (self.seasonal_components['weekly']['ds'] - date).abs().argmin()
            ]['weekly']
            
            multipliers.append({
                'date': date,
                'yearly_multiplier': yearly_mult,
                'weekly_multiplier': weekly_mult,
                'combined_multiplier': yearly_mult + weekly_mult
            })
        
        return multipliers


def train_all_models(sales_data, calendar_data, prices_data=None, progress_callback=None):
    """Train all ML models and return results with progress tracking"""
    
    results = {}
    total_models = 5  # 1 spike classifier + 3 volume regressors + 1 seasonality analyzer
    current_model = 0
    
    def update_progress(message):
        nonlocal current_model
        try:
            if progress_callback:
                progress_callback(current_model / total_models, message)
            else:
                print(f"[{current_model}/{total_models}] {message}")
        except Exception as e:
            print(f"[{current_model}/{total_models}] {message}")  # Fallback to console
    
    # 1. Demand Spike Classifier
    update_progress("Training Demand Spike Classifier...")
    spike_classifier = DemandSpikeClassifier()
    spike_results = spike_classifier.train(sales_data, calendar_data)
    results['spike_classifier'] = {
        'model': spike_classifier,
        'results': spike_results
    }
    current_model += 1
    update_progress("Demand Spike Classifier completed ✓")
    
    # 2. Volume Regressors
    for model_type in ['xgboost', 'lightgbm', 'random_forest']:
        update_progress(f"Training {model_type.title()} Volume Regressor...")
        volume_regressor = DemandVolumeRegressor(model_type)
        volume_results = volume_regressor.train(sales_data, calendar_data, prices_data)
        results[f'volume_regressor_{model_type}'] = {
            'model': volume_regressor,
            'results': volume_results
        }
        current_model += 1
        update_progress(f"{model_type.title()} Volume Regressor completed ✓")
    
    # 3. Seasonality Analyzer
    update_progress("Performing Seasonality Analysis...")
    seasonality_analyzer = SeasonalityAnalyzer()
    seasonality_results = seasonality_analyzer.analyze_seasonality(sales_data, calendar_data)
    results['seasonality_analyzer'] = {
        'model': seasonality_analyzer,
        'results': seasonality_results
    }
    current_model += 1
    update_progress("All models training completed! 🎉")
    
    return results


def generate_ml_insights(trained_models, sales_data, calendar_data, prices_data=None):
    """Generate actionable insights from trained models"""
    
    insights = {}
    
    # 1. Demand Spike Predictions
    if 'spike_classifier' in trained_models:
        try:
            spike_model = trained_models['spike_classifier']['model']
            spike_predictions = spike_model.predict(sales_data, calendar_data)
            
            # Get spike indices
            spike_indices = spike_predictions['predictions'] == 1
            drop_indices = spike_predictions['predictions'] == 2
            
            # Handle date extraction safely - we need to align indices
            spike_dates = []
            drop_dates = []
            
            if 'd' in sales_data.columns:
                # Raw M5 format - use 'd' column with boolean indexing
                spike_dates = sales_data.loc[spike_indices, 'd'].tolist() if spike_indices.any() else []
                drop_dates = sales_data.loc[drop_indices, 'd'].tolist() if drop_indices.any() else []
            elif 'date' in sales_data.columns:
                # Processed format with explicit date column
                spike_dates = sales_data.loc[spike_indices, 'date'].tolist() if spike_indices.any() else []
                drop_dates = sales_data.loc[drop_indices, 'date'].tolist() if drop_indices.any() else []
            else:
                # Use index positions converted to dates
                date_range = pd.date_range(start='2011-01-29', periods=len(sales_data))
                spike_dates = date_range[spike_indices].tolist()
                drop_dates = date_range[drop_indices].tolist()
            
            insights['demand_spikes'] = {
                'upcoming_spikes': np.sum(spike_predictions['predictions'] == 1),
                'upcoming_drops': np.sum(spike_predictions['predictions'] == 2),
                'spike_dates': spike_dates,
                'drop_dates': drop_dates
            }
        except Exception as e:
            print(f"Warning: Could not generate spike insights: {e}")
            insights['demand_spikes'] = {
                'upcoming_spikes': 0,
                'upcoming_drops': 0,
                'spike_dates': [],
                'drop_dates': []
            }
    
    # 2. Volume Predictions Comparison
    volume_predictions = {}
    for model_name, model_data in trained_models.items():
        if 'volume_regressor' in model_name:
            model = model_data['model']
            predictions = model.predict(sales_data, calendar_data, prices_data)
            volume_predictions[model_name] = predictions['predictions']
    
    if volume_predictions:
        # Ensemble prediction (average of all models)
        ensemble_pred = np.mean(list(volume_predictions.values()), axis=0)
        insights['volume_predictions'] = {
            'individual_models': volume_predictions,
            'ensemble_prediction': ensemble_pred,
            'prediction_variance': np.var(list(volume_predictions.values()), axis=0)
        }
    
    # 3. Seasonality Insights
    if 'seasonality_analyzer' in trained_models:
        try:
            seasonality_model = trained_models['seasonality_analyzer']['model']
            
            # Determine the last date safely
            if 'date' in sales_data.columns:
                last_date = pd.to_datetime(sales_data['date'].max())
            elif 'd' in sales_data.columns:
                # For M5 data, convert 'd' to actual dates
                max_d = sales_data['d'].max()
                # M5 dataset starts from 2011-01-29
                last_date = pd.to_datetime('2011-01-29') + pd.Timedelta(days=int(max_d.replace('d_', '')) - 1)
            else:
                # Use index position to estimate date
                last_date = pd.to_datetime('2011-01-29') + pd.Timedelta(days=len(sales_data) - 1)
            
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=30,
                freq='D'
            )
            seasonal_multipliers = seasonality_model.get_seasonal_multipliers(future_dates)
            
            insights['seasonality'] = {
                'future_multipliers': seasonal_multipliers,
                'analysis_results': trained_models['seasonality_analyzer']['results']
            }
        except Exception as e:
            print(f"Warning: Could not generate seasonality insights: {e}")
            insights['seasonality'] = {
                'future_multipliers': [],
                'analysis_results': {}
            }
    
    return insights
