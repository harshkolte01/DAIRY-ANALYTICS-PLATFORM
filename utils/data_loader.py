import pandas as pd
import numpy as np
import os

def load_sales_data():
    path = "data/sales_train_validation.csv"
    return pd.read_csv(path)

def load_calendar_data():
    return pd.read_csv("data/calendar.csv")

def load_prices_data():
    """Load selling prices data"""
    return pd.read_csv("data/sell_prices.csv")

def get_item_pricing_data(prices_df, calendar_df, item_id, store_id=None):
    """
    Get pricing data for a specific item, optionally filtered by store
    Maps weekly pricing data to daily dates
    """
    # Filter for specific item
    item_prices = prices_df[prices_df['item_id'] == item_id].copy()
    
    if store_id:
        item_prices = item_prices[item_prices['store_id'] == store_id]
    
    if item_prices.empty:
        return pd.DataFrame()
    
    # Merge with calendar to get date mappings
    calendar_subset = calendar_df[['date', 'wm_yr_wk']].copy()
    calendar_subset['date'] = pd.to_datetime(calendar_subset['date'])
    
    # Merge pricing data with calendar
    pricing_data = pd.merge(item_prices, calendar_subset, on='wm_yr_wk', how='inner')
    
    # Sort by date
    pricing_data = pricing_data.sort_values('date').reset_index(drop=True)
    
    return pricing_data[['date', 'store_id', 'item_id', 'sell_price', 'wm_yr_wk']]

def calculate_average_price_by_plant(prices_df, calendar_df, item_id, store_ids):
    """
    Calculate average selling price for an item across multiple plants/stores
    """
    plant_pricing = {}
    
    for store_id in store_ids:
        pricing_data = get_item_pricing_data(prices_df, calendar_df, item_id, store_id)
        if not pricing_data.empty:
            avg_price = pricing_data['sell_price'].mean()
            price_volatility = pricing_data['sell_price'].std()
            min_price = pricing_data['sell_price'].min()
            max_price = pricing_data['sell_price'].max()
            
            plant_pricing[store_id] = {
                'avg_price': avg_price,
                'price_volatility': price_volatility,
                'min_price': min_price,
                'max_price': max_price,
                'price_data': pricing_data
            }
    
    return plant_pricing

def simulate_production_costs(plant_capacities, base_cost_per_unit=0.60):
    """
    Simulate production costs based on plant capacity and operational efficiency
    Larger plants typically have lower per-unit costs due to economies of scale
    """
    production_costs = {}
    
    for plant_id, capacity in plant_capacities.items():
        # Base cost varies with plant size (economies of scale)
        if capacity >= 1000:  # Large plants
            cost_multiplier = 0.85  # 15% cost reduction
        elif capacity >= 800:   # Medium plants
            cost_multiplier = 0.95  # 5% cost reduction
        else:                   # Small plants
            cost_multiplier = 1.1   # 10% cost increase
        
        # Add some plant-specific variation
        plant_variation = np.random.uniform(0.95, 1.05)
        
        production_costs[plant_id] = {
            'cost_per_unit': base_cost_per_unit * cost_multiplier * plant_variation,
            'fixed_daily_cost': capacity * 0.1,  # Fixed costs scale with capacity
            'capacity': capacity
        }
    
    return production_costs

def preprocess_sales_data(sales_df, calendar_df):
    # Example: Aggregate one SKU for forecasting
    sku = sales_df[sales_df['item_id'] == 'FOODS_3_090']
    
    # Sum across all stores/states for this item to get total sales per day
    day_columns = sku.iloc[:, 6:]  # Only day columns
    daily_sales = day_columns.sum(axis=0)  # Sum across all rows (stores/states)
    
    # Create time series DataFrame
    ts = pd.DataFrame({
        'ds': pd.date_range(start='2011-01-29', periods=len(daily_sales)),
        'sales': daily_sales.values
    })
    
    return ts

def get_available_stores_and_states(sales_df):
    """
    Get unique store_ids and their corresponding state_ids for multi-plant analysis
    Returns a simple dictionary mapping store_id to state_id
    """
    # Get unique store-state combinations
    store_state_df = sales_df[['store_id', 'state_id']].drop_duplicates()
    
    # Create a simple dictionary mapping store_id to state_id
    store_to_state = dict(zip(store_state_df['store_id'], store_state_df['state_id']))
    
    return store_to_state

def preprocess_sales_data_by_plant(sales_df, calendar_df, store_id, item_id='FOODS_3_090'):
    """
    Preprocess sales data for a specific plant (store_id)
    
    Parameters:
    - sales_df: Sales data
    - calendar_df: Calendar data
    - store_id: Specific store_id to analyze
    - item_id: Item to analyze
    """
    # Filter for specific item and store
    plant_data = sales_df[(sales_df['item_id'] == item_id) & (sales_df['store_id'] == store_id)].copy()
    
    if plant_data.empty:
        return pd.DataFrame()  # Return empty DataFrame if no data
        
    # Get day columns (sales data)
    day_columns = plant_data.iloc[:, 6:]  # Only day columns
    daily_sales = day_columns.sum(axis=0)  # Sum across all rows
    
    # Create time series DataFrame
    ts = pd.DataFrame({
        'ds': pd.date_range(start='2011-01-29', periods=len(daily_sales)),
        'sales': daily_sales.values
    })
    
    return ts

def get_plant_capacity_mapping(store_ids, store_to_state_mapping):
    """
    Get capacity mapping for specific store IDs
    Uses simulated capacity based on state and store patterns
    """
    plant_capacities = {}
    
    for store_id in store_ids:
        state = store_to_state_mapping.get(store_id, 'Unknown')
        
        # Base capacity by state (simulated based on typical dairy regions)
        if state == 'CA':
            base_capacity = np.random.randint(900, 1200)  # California - large operations
        elif state == 'TX':
            base_capacity = np.random.randint(750, 900)   # Texas - medium operations
        elif state == 'WI':
            base_capacity = np.random.randint(850, 1000)  # Wisconsin - traditional dairy
        else:
            base_capacity = np.random.randint(600, 900)   # Other states
        
        plant_capacities[store_id] = base_capacity
    
    return plant_capacities

def simulate_milk_supply(calendar_df, base_supply=800, seasonal_factor=0.2, event_factor=0.15):
    """
    Simulate milk supply based on seasonal trends and calendar events.
    
    Parameters:
    - calendar_df: Calendar data with dates, months, and events
    - base_supply: Base daily milk supply capacity
    - seasonal_factor: How much seasonal variation affects supply (0-1)
    - event_factor: How much events affect supply (0-1)
    """
    supply_data = []
    
    for _, row in calendar_df.iterrows():
        date = pd.to_datetime(row['date'])
        month = row['month']
        
        # Base supply
        daily_supply = base_supply
        
        # Seasonal adjustments (milk production patterns)
        seasonal_multiplier = get_seasonal_multiplier(month, seasonal_factor)
        daily_supply *= seasonal_multiplier
        
        # Event-based adjustments
        event_multiplier = get_event_multiplier(row, event_factor)
        daily_supply *= event_multiplier
        
        # Add some random variation (±5%)
        daily_supply *= np.random.uniform(0.95, 1.05)
        
        supply_data.append({
            'ds': date,
            'milk_supply': max(0, daily_supply),  # Ensure non-negative
            'seasonal_factor': seasonal_multiplier,
            'event_factor': event_multiplier
        })
    
    return pd.DataFrame(supply_data)

def get_seasonal_multiplier(month, seasonal_factor):
    """
    Get seasonal multiplier for milk supply based on month.
    Higher in cooler months, lower in hot summer months.
    """
    # Seasonal pattern: higher supply in cooler months (Oct-Mar), lower in summer (Jun-Aug)
    seasonal_pattern = {
        1: 1.1,   # January - winter, good supply
        2: 1.05,  # February
        3: 1.0,   # March - spring starts
        4: 0.95,  # April
        5: 0.9,   # May - getting warmer
        6: 0.8,   # June - summer heat affects cows
        7: 0.75,  # July - peak summer, lowest supply
        8: 0.8,   # August - still hot
        9: 0.9,   # September - cooling down
        10: 1.0,  # October - good weather
        11: 1.05, # November - cooler
        12: 1.1   # December - winter
    }
    
    base_multiplier = seasonal_pattern.get(month, 1.0)
    # Apply seasonal factor (how much seasonality affects supply)
    return 1.0 + (base_multiplier - 1.0) * seasonal_factor

def get_event_multiplier(row, event_factor):
    """
    Get event-based multiplier for milk supply.
    Events like festivals increase demand but may disrupt supply logistics.
    """
    event_name_1 = str(row.get('event_name_1', '')).lower()
    event_type_1 = str(row.get('event_type_1', '')).lower()
    event_name_2 = str(row.get('event_name_2', '')).lower()
    event_type_2 = str(row.get('event_type_2', '')).lower()
    
    multiplier = 1.0
    
    # Major holidays/festivals - supply chain disruptions
    major_events = ['christmas', 'thanksgiving', 'easter', 'newyear', 'independenceday']
    if any(event in event_name_1 for event in major_events) or \
       any(event in event_name_2 for event in major_events):
        multiplier *= (1.0 - event_factor * 0.8)  # 80% of event factor reduction
    
    # Cultural/Religious events - moderate impact
    elif 'cultural' in event_type_1 or 'religious' in event_type_1 or \
         'cultural' in event_type_2 or 'religious' in event_type_2:
        multiplier *= (1.0 - event_factor * 0.5)  # 50% of event factor reduction
    
    # National holidays - supply chain slowdown
    elif 'national' in event_type_1 or 'national' in event_type_2:
        multiplier *= (1.0 - event_factor * 0.6)  # 60% of event factor reduction
    
    # Sporting events - usually increase demand, slight supply pressure
    elif 'sporting' in event_type_1 or 'sporting' in event_type_2:
        multiplier *= (1.0 - event_factor * 0.2)  # 20% of event factor reduction
    
    # Weekend effect - slightly reduced supply due to reduced farm operations
    weekday = str(row.get('weekday', '')).lower()
    if weekday in ['saturday', 'sunday']:
        multiplier *= 0.95
    
    return max(0.5, multiplier)  # Ensure supply doesn't drop below 50% of base
