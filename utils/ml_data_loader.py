"""
ML-Enhanced Data Loader for Dairy Analytics Platform
Integrates ML models and advanced features with existing data loading functionality
"""

import pandas as pd
import numpy as np
import os
from .ml_models import train_all_models, generate_ml_insights
from .feature_engineering import create_comprehensive_features, AdvancedFeatureEngineer
import pickle
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Import existing functions
from .data_loader import (
    load_sales_data, load_calendar_data, load_prices_data,
    preprocess_sales_data, get_available_stores_and_states
)

class MLEnhancedDataLoader:
    """
    Enhanced data loader with ML capabilities and advanced features
    """
    
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.trained_models = {}
        self.feature_engineer = None
        self.ml_insights = {}
        self.is_trained = False
        
    def load_and_prepare_data(self, store_id=None, item_id=None):
        """Load and prepare data with ML enhancements"""
        
        # Load base data
        sales_df = load_sales_data()
        calendar_df = load_calendar_data()
        prices_df = load_prices_data()
        
        # Filter if specified
        if store_id:
            sales_df = sales_df[sales_df['store_id'] == store_id]
        if item_id:
            sales_df = sales_df[sales_df['item_id'] == item_id]
        
        # Preprocess
        ts_data = preprocess_sales_data(sales_df, calendar_df)
        
        return {
            'sales_df': sales_df,
            'calendar_df': calendar_df,
            'prices_df': prices_df,
            'ts_data': ts_data
        }
    
    def train_ml_models(self, store_id=None, item_id=None, save_models=True, use_streamlit=True):
        """Train all ML models on the data with progress tracking"""
        
        start_time = time.time()
        total_steps = 6  # Data loading, feature engineering, 3 models, insights, saving
        
        # Initialize progress tracking variables
        progress_bar = None
        status_text = None
        time_text = None
        
        if use_streamlit and STREAMLIT_AVAILABLE:
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
        
        def update_progress(step, step_name, elapsed_time):
            progress = step / total_steps
            if use_streamlit and STREAMLIT_AVAILABLE and progress_bar is not None:
                progress_bar.progress(progress)
                status_text.text(f"Step {step}/{total_steps}: {step_name}")
                
                # Calculate ETA
                if step > 0:
                    avg_time_per_step = elapsed_time / step
                    remaining_steps = total_steps - step
                    eta_seconds = avg_time_per_step * remaining_steps
                    eta = timedelta(seconds=int(eta_seconds))
                    time_text.text(f"⏱️ Elapsed: {timedelta(seconds=int(elapsed_time))} | ETA: {eta}")
                else:
                    time_text.text(f"⏱️ Elapsed: {timedelta(seconds=int(elapsed_time))}")
            else:
                # Console progress for standalone training
                console_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
                if step > 0:
                    avg_time_per_step = elapsed_time / step
                    remaining_steps = total_steps - step
                    eta_seconds = avg_time_per_step * remaining_steps
                    eta = timedelta(seconds=int(eta_seconds))
                    print(f"\r[{console_bar}] {progress*100:.1f}% | {step_name} | Elapsed: {timedelta(seconds=int(elapsed_time))} | ETA: {eta}", end="", flush=True)
                else:
                    print(f"\r[{console_bar}] {progress*100:.1f}% | {step_name} | Elapsed: {timedelta(seconds=int(elapsed_time))}", end="", flush=True)
        
        try:
            # Step 1: Load and prepare data
            update_progress(0, "Loading and preparing data...", time.time() - start_time)
            data = self.load_and_prepare_data(store_id, item_id)
            update_progress(1, "Data loaded ✓", time.time() - start_time)
            
            # Step 2: Create comprehensive features
            update_progress(1, "Creating advanced features...", time.time() - start_time)
            feature_data = create_comprehensive_features(
                data['sales_df'], 
                data['calendar_df'], 
                data['prices_df']
            )
            self.feature_engineer = feature_data['feature_engineer']
            update_progress(2, "Features created ✓", time.time() - start_time)
            
            # Step 3-5: Train models (this function handles internal progress)
            update_progress(2, "Training ML models (this may take 3-5 minutes)...", time.time() - start_time)
            self.trained_models = train_all_models(
                data['ts_data'], 
                data['calendar_df'], 
                data['prices_df'],
                progress_callback=lambda p, msg: update_progress(2 + p*3, f"Training: {msg}", time.time() - start_time) if use_streamlit and STREAMLIT_AVAILABLE else None
            )
            update_progress(5, "Models trained ✓", time.time() - start_time)
            
            # Step 6: Generate insights
            update_progress(5, "Generating ML insights...", time.time() - start_time)
            self.ml_insights = generate_ml_insights(
                self.trained_models,
                data['ts_data'],
                data['calendar_df'],
                data['prices_df']
            )
            
            # Add feature insights
            self.ml_insights.update({
                'feature_insights': feature_data['insights'],
                'anomalies': feature_data['anomalies'],
                'recommendations': feature_data['recommendations']
            })
            
            self.is_trained = True
            update_progress(6, "Training completed ✓", time.time() - start_time)
            
            # Save models if requested
            if save_models:
                try:
                    self.save_models()
                except Exception as save_error:
                    print(f"Warning: Could not save models: {save_error}")
            
            return {
                'training_results': {name: data['results'] for name, data in self.trained_models.items()},
                'insights': self.ml_insights
            }
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            if use_streamlit and STREAMLIT_AVAILABLE and status_text is not None:
                status_text.text(f"❌ {error_msg}")
            else:
                print(f"\n❌ {error_msg}")
            raise e
    
    def get_demand_predictions(self, days_ahead=30):
        """Get ML-enhanced demand predictions"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained first. Call train_ml_models()")
        
        predictions = {}
        
        # Get volume predictions from different models
        if 'volume_regressor_xgboost' in self.trained_models:
            xgb_model = self.trained_models['volume_regressor_xgboost']['model']
            # Note: This is simplified - you'd need to create future features
            predictions['xgboost'] = "Model available - implement future feature creation"
        
        # Get spike predictions
        if 'spike_classifier' in self.trained_models:
            spike_model = self.trained_models['spike_classifier']['model']
            predictions['spike_alerts'] = "Model available - implement future feature creation"
        
        return predictions
    
    def get_business_insights(self):
        """Get comprehensive business insights from ML analysis"""
        
        if not self.is_trained:
            return {"message": "Train ML models first to get insights"}
        
        return self.ml_insights
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        
        if not self.is_trained:
            return {"message": "Train ML models first"}
        
        importance_data = {}
        
        for model_name, model_data in self.trained_models.items():
            if 'results' in model_data and 'feature_importance' in model_data['results']:
                importance_data[model_name] = model_data['results']['feature_importance']
        
        return importance_data
    
    def save_models(self, filepath='models/trained_models.pkl'):
        """Save trained models to disk"""
        
        if not self.is_trained:
            print("No models to save. Train models first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save models and related data
        save_data = {
            'trained_models': self.trained_models,
            'feature_engineer': self.feature_engineer,
            'ml_insights': self.ml_insights,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='models/trained_models.pkl'):
        """Load trained models from disk"""
        
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.trained_models = save_data['trained_models']
            self.feature_engineer = save_data['feature_engineer']
            self.ml_insights = save_data['ml_insights']
            self.is_trained = save_data['is_trained']
            
            print(f"Models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def analyze_store_performance(self, store_ids=None):
        """Analyze performance across multiple stores using ML"""
        
        if store_ids is None:
            # Get available stores
            sales_df = load_sales_data()
            stores_states = get_available_stores_and_states(sales_df)
            store_ids = list(stores_states.keys())[:5]  # Analyze top 5 stores
        
        store_analysis = {}
        
        for store_id in store_ids:
            print(f"Analyzing store {store_id}...")
            
            # Load data for this store
            data = self.load_and_prepare_data(store_id=store_id)
            
            # Create features
            feature_data = create_comprehensive_features(
                data['sales_df'], 
                data['calendar_df'], 
                data['prices_df']
            )
            
            store_analysis[store_id] = {
                'insights': feature_data['insights'],
                'anomalies': feature_data['anomalies'],
                'recommendations': feature_data['recommendations'],
                'data_quality': {
                    'total_records': len(data['sales_df']),
                    'date_range': f"{data['ts_data']['ds'].min()} to {data['ts_data']['ds'].max()}",
                    'avg_daily_sales': data['ts_data']['sales'].mean(),
                    'sales_volatility': data['ts_data']['sales'].std()
                }
            }
        
        return store_analysis


def create_ml_enhanced_forecast(sales_data, calendar_data, prices_data=None, ml_enhanced=True):
    """Enhanced forecasting with ML features"""
    
    if not ml_enhanced:
        # Fall back to regular Prophet forecasting
        from .forecast import forecast_demand
        return forecast_demand(sales_data)
    
    # Create ML-enhanced forecast
    ml_loader = MLEnhancedDataLoader()
    
    # Create comprehensive features
    feature_data = create_comprehensive_features(sales_data, calendar_data, prices_data)
    
    # For now, return the original forecast with ML insights attached
    from .forecast import forecast_demand
    base_forecast = forecast_demand(sales_data)
    
    # Add ML insights to forecast
    base_forecast['ml_insights'] = feature_data['insights']
    base_forecast['ml_recommendations'] = feature_data['recommendations']
    base_forecast['feature_count'] = len(feature_data['features'].columns)
    
    return base_forecast


def get_ml_model_summary():
    """Get summary of available ML models and their capabilities"""
    
    return {
        'models': {
            'demand_spike_classifier': {
                'purpose': 'Predict demand spikes and drops',
                'input': 'Time series features, events, prices',
                'output': 'Spike/Normal/Drop classification',
                'business_value': 'Proactive inventory management'
            },
            'volume_regressor_xgboost': {
                'purpose': 'Predict exact demand volumes',
                'input': 'Advanced engineered features',
                'output': 'Numerical demand prediction',
                'business_value': 'Accurate production planning'
            },
            'volume_regressor_lightgbm': {
                'purpose': 'Alternative volume prediction',
                'input': 'Advanced engineered features',
                'output': 'Numerical demand prediction',
                'business_value': 'Model ensemble for robustness'
            },
            'seasonality_analyzer': {
                'purpose': 'Decompose seasonal patterns',
                'input': 'Historical time series',
                'output': 'Seasonal components and multipliers',
                'business_value': 'Long-term capacity planning'
            }
        },
        'features': {
            'time_features': 'Day of week, month, seasonality cycles',
            'lag_features': 'Historical values at different time lags',
            'rolling_features': 'Moving averages, trends, volatility',
            'price_features': 'Price changes, relative pricing, volatility',
            'event_features': 'SNAP events, holidays, special occasions',
            'statistical_features': 'Historical patterns by item/store',
            'interaction_features': 'Combined effects of multiple variables'
        },
        'insights': {
            'demand_drivers': 'Key factors influencing demand',
            'price_sensitivity': 'Impact of pricing on sales',
            'event_impact': 'Effect of events on demand patterns',
            'seasonality': 'Monthly and weekly demand patterns',
            'anomalies': 'Unusual demand patterns and outliers',
            'recommendations': 'Actionable business recommendations'
        }
    }


# Enhanced functions that can be imported by app.py
def load_ml_enhanced_data():
    """Load data with ML enhancements ready"""
    
    ml_loader = MLEnhancedDataLoader()
    return ml_loader

def get_advanced_analytics_summary(sales_df, calendar_df, prices_df=None):
    """Get advanced analytics summary without full ML training"""
    
    # Quick feature analysis
    feature_data = create_comprehensive_features(sales_df, calendar_df, prices_df)
    
    return {
        'feature_count': len(feature_data['features'].columns),
        'insights': feature_data['insights'],
        'recommendations': feature_data['recommendations'][:5],  # Top 5 recommendations
        'anomaly_percentage': feature_data['anomalies'].get('statistical_outliers', {}).get('percentage', 0)
    }
