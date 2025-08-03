from prophet import Prophet

def forecast_demand(ts_df, days=28):
    """
    Forecast demand using Prophet
    Returns forecast DataFrame with historical and predicted values
    """
    # Prepare data for Prophet (rename columns)
    df = ts_df.rename(columns={'ds': 'ds', 'sales': 'y'})
    
    # Fit Prophet model
    model = Prophet()
    model.fit(df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=days)
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Merge with original data to include historical sales
    forecast_with_history = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # Add historical sales data
    forecast_with_history = forecast_with_history.merge(
        ts_df[['ds', 'sales']], 
        on='ds', 
        how='left'
    )
    
    # For historical data, use actual sales; for future, use predictions
    forecast_with_history['y'] = forecast_with_history['sales'].fillna(forecast_with_history['yhat'])
    
    return forecast_with_history
