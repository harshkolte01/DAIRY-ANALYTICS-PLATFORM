from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpMaximize
import pandas as pd
import numpy as np

def optimize_for_profit(forecast_df, pricing_data, production_costs, plant_capacity, supply_df=None):
    """
    Optimize production schedule for maximum profit instead of just capacity utilization
    Considers revenue (price * quantity) - production costs (variable + fixed)
    """
    try:
        # Handle different column names for demand data
        if 'yhat' in forecast_df.columns:
            demand_col = 'yhat'
        elif 'y' in forecast_df.columns:
            demand_col = 'y'
        elif 'sales' in forecast_df.columns:
            demand_col = 'sales'
        else:
            raise ValueError("No valid demand column found in forecast_df")
        
        # Profit maximization problem
        prob = LpProblem("Dairy_Profit_Optimization", LpMaximize)

        days = forecast_df['ds'].astype(str).tolist()
        demand = forecast_df[demand_col].tolist()
        
        # Ensure demand is non-negative (forecast can produce negative values)
        demand = [max(0, d) for d in demand]
        
        # Get pricing data for each day
        pricing_dict = {}
        if not pricing_data.empty:
            pricing_data['date_str'] = pricing_data['date'].astype(str)
            for _, row in pricing_data.iterrows():
                pricing_dict[row['date_str']] = row['sell_price']
        
        # Default price if not found
        default_price = pricing_data['sell_price'].mean() if not pricing_data.empty else 1.0
        daily_prices = [pricing_dict.get(day, default_price) for day in days]
        
        # Production cost parameters
        variable_cost = production_costs['cost_per_unit']
        fixed_daily_cost = production_costs['fixed_daily_cost']
        
        # Get supply constraints for each day
        if supply_df is not None:
            merged_df = pd.merge(forecast_df, supply_df, on='ds', how='left')
            supply_capacity = merged_df['milk_supply'].fillna(plant_capacity).tolist()
        else:
            supply_capacity = [plant_capacity] * len(days)

        # Decision variables
        prod_vars = {d: LpVariable(f'prod_{d}', lowBound=0) for d in days}
        sales_vars = {d: LpVariable(f'sales_{d}', lowBound=0) for d in days}
        
        # Objective: Maximize total profit
        total_revenue = lpSum([sales_vars[d] * daily_prices[i] for i, d in enumerate(days)])
        total_variable_cost = lpSum([prod_vars[d] * variable_cost for d in days])
        total_fixed_cost = fixed_daily_cost * len(days)
        
        prob += total_revenue - total_variable_cost - total_fixed_cost
        
        # Constraints
        for i, day in enumerate(days):
            # Production capacity constraint
            prob += prod_vars[day] <= supply_capacity[i], f"Capacity_{day}"
            
            # Sales cannot exceed production
            prob += sales_vars[day] <= prod_vars[day], f"Sales_Production_{day}"
            
            # Sales cannot exceed demand
            prob += sales_vars[day] <= demand[i], f"Sales_Demand_{day}"
        
        # Solve the problem
        prob.solve()
        
        if prob.status != 1:  # Not optimal
            return None
        
        # Extract results
        results = []
        for i, day in enumerate(days):
            production = prod_vars[day].varValue or 0
            sales = sales_vars[day].varValue or 0
            price = daily_prices[i]
            
            revenue = sales * price
            variable_cost_day = production * variable_cost
            profit = revenue - variable_cost_day - (fixed_daily_cost / len(days))
            
            results.append({
                'ds': pd.to_datetime(day),
                'forecast_demand': demand[i],
                'optimal_production': production,
                'actual_sales': sales,
                'sell_price': price,
                'revenue': revenue,
                'variable_cost': variable_cost_day,
                'daily_profit': profit,
                'supply_capacity': supply_capacity[i],
                'demand_fulfillment': (sales / demand[i] * 100) if demand[i] > 0 else 100,
                'capacity_utilization': (production / supply_capacity[i] * 100) if supply_capacity[i] > 0 else 0,
                'profit_margin': (profit / revenue * 100) if revenue > 0 else 0
            })
        
        result_df = pd.DataFrame(results)
        return result_df
        
    except Exception as e:
        print(f"Profit optimization failed: {str(e)}")
        return None

def optimize_schedule(forecast_df, supply_df=None, base_capacity=1000):
    """
    Optimize production schedule to meet demand while respecting supply constraints.
    Uses linear programming to minimize the difference between production and demand.
    """
    try:
        # Handle different column names for demand data
        if 'yhat' in forecast_df.columns:
            demand_col = 'yhat'
        elif 'y' in forecast_df.columns:
            demand_col = 'y'
        elif 'sales' in forecast_df.columns:
            demand_col = 'sales'
        else:
            raise ValueError("No valid demand column found in forecast_df")
        
        # Simple linear optimization: minimize absolute difference from forecast under capacity constraint
        prob = LpProblem("Dairy_Production", LpMinimize)

        days = forecast_df['ds'].astype(str).tolist()
        demand = forecast_df[demand_col].tolist()
        
        # Get supply constraints for each day
        if supply_df is not None:
            # Merge forecast with supply data
            merged_df = pd.merge(forecast_df, supply_df, on='ds', how='left')
            supply_capacity = merged_df['milk_supply'].fillna(base_capacity).tolist()
        else:
            supply_capacity = [base_capacity] * len(days)

        prod_vars = {d: LpVariable(f'prod_{d}', lowBound=0) for d in days}
        
        # Create auxiliary variables for absolute differences
        pos_diff = {d: LpVariable(f'pos_diff_{d}', lowBound=0) for d in days}
        neg_diff = {d: LpVariable(f'neg_diff_{d}', lowBound=0) for d in days}
        
        # Additional variables for supply-demand mismatch analysis
        supply_shortage = {d: LpVariable(f'shortage_{d}', lowBound=0) for d in days}

        # Multi-objective: minimize demand mismatch + penalize supply shortages
        prob += lpSum([pos_diff[d] + neg_diff[d] + 2 * supply_shortage[d] for d in days])

        # Constraints for absolute difference
        for i, d in enumerate(days):
            prob += prod_vars[d] - demand[i] <= pos_diff[d]
            prob += demand[i] - prod_vars[d] <= neg_diff[d]
            
            # Supply capacity constraint
            prob += prod_vars[d] <= supply_capacity[i]
            
            # Supply shortage tracking
            prob += demand[i] - supply_capacity[i] <= supply_shortage[d]

        prob.solve()

        result = pd.DataFrame({
            "date": days,
            "forecast_demand": demand,
            "supply_capacity": supply_capacity,
            "optimal_production": [prod_vars[d].varValue for d in days],
            "supply_utilization": [
                (prod_vars[d].varValue / supply_capacity[i]) * 100 
                if supply_capacity[i] > 0 else 0 
                for i, d in enumerate(days)
            ],
            "demand_fulfillment": [
                min(100, (prod_vars[d].varValue / demand[i]) * 100) 
                if demand[i] > 0 else 100 
                for i, d in enumerate(days)
            ]
        })
        
        return result
        
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        # Fallback to basic production schedule
        return create_basic_schedule(forecast_df, supply_df, base_capacity)

def create_basic_schedule(forecast_df, supply_df=None, base_capacity=1000):
    """
    Create a basic production schedule without optimization as a fallback.
    """
    days = forecast_df['ds'].astype(str).tolist()
    demand = forecast_df['yhat'].tolist()
    
    # Get supply constraints for each day
    if supply_df is not None:
        merged_df = pd.merge(forecast_df, supply_df, on='ds', how='left')
        supply_capacity = merged_df['milk_supply'].fillna(base_capacity).tolist()
    else:
        supply_capacity = [base_capacity] * len(days)
    
    # Simple heuristic: produce minimum of demand and capacity
    optimal_production = [min(demand[i], supply_capacity[i]) for i in range(len(days))]
    
    result = pd.DataFrame({
        "date": days,
        "forecast_demand": demand,
        "supply_capacity": supply_capacity,
        "optimal_production": optimal_production,
        "supply_utilization": [
            (optimal_production[i] / supply_capacity[i]) * 100 
            if supply_capacity[i] > 0 else 0 
            for i in range(len(days))
        ],
        "demand_fulfillment": [
            min(100, (optimal_production[i] / demand[i]) * 100) 
            if demand[i] > 0 else 100 
            for i in range(len(days))
        ]
    })
    
    return result

def analyze_supply_demand_gaps(forecast_df, supply_df):
    """
    Analyze gaps between supply capacity and demand forecasts.
    """
    merged_df = pd.merge(forecast_df, supply_df, on='ds', how='inner')
    
    merged_df['supply_demand_gap'] = merged_df['milk_supply'] - merged_df['yhat']
    merged_df['gap_percentage'] = (merged_df['supply_demand_gap'] / merged_df['yhat']) * 100
    
    # Categorize days
    merged_df['status'] = np.where(
        merged_df['supply_demand_gap'] >= 0, 'Surplus', 'Shortage'
    )
    
    summary = {
        'total_days': len(merged_df),
        'surplus_days': len(merged_df[merged_df['status'] == 'Surplus']),
        'shortage_days': len(merged_df[merged_df['status'] == 'Shortage']),
        'avg_surplus': merged_df[merged_df['supply_demand_gap'] > 0]['supply_demand_gap'].mean(),
        'avg_shortage': abs(merged_df[merged_df['supply_demand_gap'] < 0]['supply_demand_gap'].mean()),
        'max_shortage': abs(merged_df['supply_demand_gap'].min()),
        'max_surplus': merged_df['supply_demand_gap'].max()
    }
    
    return merged_df, summary

def get_optimization_summary(optimization_result):
    """
    Generate a summary of the optimization results.
    """
    if optimization_result is None or optimization_result.empty:
        return {
            "status": "Failed",
            "total_production": 0,
            "total_demand": 0,
            "avg_utilization": 0,
            "avg_fulfillment": 0,
            "shortage_days": 0,
            "surplus_days": 0
        }
    
    total_production = optimization_result['optimal_production'].sum()
    total_demand = optimization_result['forecast_demand'].sum()
    avg_utilization = optimization_result['supply_utilization'].mean()
    avg_fulfillment = optimization_result['demand_fulfillment'].mean()
    
    shortage_days = len(optimization_result[optimization_result['demand_fulfillment'] < 100])
    surplus_days = len(optimization_result[optimization_result['supply_utilization'] < 90])
    
    return {
        "status": "Success",
        "total_production": round(total_production, 2),
        "total_demand": round(total_demand, 2),
        "avg_utilization": round(avg_utilization, 2),
        "avg_fulfillment": round(avg_fulfillment, 2),
        "shortage_days": shortage_days,
        "surplus_days": surplus_days
    }

def optimize_multi_plant_schedule(multi_plant_forecasts, plant_capacities, base_capacity=1000):
    """
    Optimize production schedules across multiple plants/regions
    
    Parameters:
    - multi_plant_forecasts: Dict of {plant_id: forecast_df}
    - plant_capacities: Dict of {plant_id: capacity}
    - base_capacity: Default capacity if plant not in capacities dict
    """
    multi_plant_results = {}
    
    for plant_id, forecast_df in multi_plant_forecasts.items():
        try:
            # Get plant-specific capacity
            plant_capacity = plant_capacities.get(plant_id, base_capacity)
            
            # Ensure forecast_df has the right columns
            if 'ds' not in forecast_df.columns:
                print(f"Warning: 'ds' column missing for plant {plant_id}")
                continue
                
            # Create supply dataframe with constant capacity for this plant
            supply_df = pd.DataFrame({
                'ds': forecast_df['ds'],
                'milk_supply': [plant_capacity] * len(forecast_df)
            })
            
            # Optimize for this plant
            result_df = optimize_schedule(forecast_df, supply_df, plant_capacity)
            multi_plant_results[plant_id] = result_df
            
        except Exception as e:
            print(f"Error optimizing plant {plant_id}: {str(e)}")
            continue
    
    return multi_plant_results

def compare_plant_performance(multi_plant_data, plant_capacities, optimization_results):
    """
    Compare performance metrics across multiple plants
    """
    comparative_metrics = {}
    
    for plant_id in optimization_results.keys():
        try:
            result_df = optimization_results[plant_id]
            plant_capacity = plant_capacities.get(plant_id, 1000)
            
            if result_df is not None and not result_df.empty:
                comparative_metrics[plant_id] = {
                    'plant_id': plant_id,
                    'capacity': plant_capacity,
                    'total_demand': result_df['forecast_demand'].sum() if 'forecast_demand' in result_df.columns else 0,
                    'total_production': result_df['optimal_production'].sum() if 'optimal_production' in result_df.columns else 0,
                    'avg_utilization': result_df['supply_utilization'].mean() if 'supply_utilization' in result_df.columns else 0,
                    'avg_fulfillment': result_df['demand_fulfillment'].mean() if 'demand_fulfillment' in result_df.columns else 0,
                    'shortage_days': len(result_df[result_df['demand_fulfillment'] < 100]) if 'demand_fulfillment' in result_df.columns else 0,
                    'efficiency_score': calculate_plant_efficiency(result_df, plant_capacity)
                }
            else:
                # Default values for failed optimization
                comparative_metrics[plant_id] = {
                    'plant_id': plant_id,
                    'capacity': plant_capacity,
                    'total_demand': 0,
                    'total_production': 0,
                    'avg_utilization': 0,
                    'avg_fulfillment': 0,
                    'shortage_days': 0,
                    'efficiency_score': 0
                }
        except Exception as e:
            print(f"Error calculating metrics for plant {plant_id}: {str(e)}")
            # Provide default values
            comparative_metrics[plant_id] = {
                'plant_id': plant_id,
                'capacity': plant_capacities.get(plant_id, 1000),
                'total_demand': 0,
                'total_production': 0,
                'avg_utilization': 0,
                'avg_fulfillment': 0,
                'shortage_days': 0,
                'efficiency_score': 0
            }
    
    return comparative_metrics

def calculate_plant_efficiency(optimization_result, plant_capacity):
    """
    Calculate overall efficiency score for a plant
    Combines utilization, fulfillment, and consistency metrics
    """
    if optimization_result.empty:
        return 0
    
    avg_utilization = optimization_result['supply_utilization'].mean()
    avg_fulfillment = optimization_result['demand_fulfillment'].mean()
    
    # Penalize very high utilization (over 95%) as it indicates potential bottlenecks
    utilization_score = min(avg_utilization, 95) / 95 * 100
    
    # Fulfillment score
    fulfillment_score = avg_fulfillment
    
    # Consistency score (penalize high variance in utilization)
    utilization_std = optimization_result['supply_utilization'].std()
    consistency_score = max(0, 100 - utilization_std)
    
    # Weighted efficiency score
    efficiency_score = (
        utilization_score * 0.4 +
        fulfillment_score * 0.4 +
        consistency_score * 0.2
    )
    
    return round(efficiency_score, 2)

def analyze_plant_capacity_allocation(comparative_metrics, total_budget=None):
    """
    Analyze current capacity allocation and suggest improvements
    
    Parameters:
    - comparative_metrics: Output from optimize_multi_plant_schedule
    - total_budget: Total capacity budget for reallocation suggestions
    """
    analysis = {
        'current_allocation': {},
        'performance_ranking': [],
        'recommendations': [],
        'reallocation_suggestions': []
    }
    
    # Current allocation analysis
    total_capacity = sum([metrics['capacity'] for metrics in comparative_metrics.values()])
    total_demand = sum([metrics['total_demand'] for metrics in comparative_metrics.values()])
    
    for plant_id, metrics in comparative_metrics.items():
        analysis['current_allocation'][plant_id] = {
            'capacity_share': round((metrics['capacity'] / total_capacity) * 100, 2),
            'demand_share': round((metrics['total_demand'] / total_demand) * 100, 2),
            'capacity_utilization': round(metrics['avg_utilization'], 2),
            'efficiency_score': metrics['efficiency_score']
        }
    
    # Performance ranking
    sorted_plants = sorted(comparative_metrics.items(), 
                          key=lambda x: x[1]['efficiency_score'], reverse=True)
    
    for rank, (plant_id, metrics) in enumerate(sorted_plants, 1):
        analysis['performance_ranking'].append({
            'rank': rank,
            'plant_id': plant_id,
            'efficiency_score': metrics['efficiency_score'],
            'avg_utilization': round(metrics['avg_utilization'], 2),
            'avg_fulfillment': round(metrics['avg_fulfillment'], 2)
        })
    
    # Generate recommendations
    for plant_id, metrics in comparative_metrics.items():
        current_alloc = analysis['current_allocation'][plant_id]
        
        if metrics['avg_utilization'] > 95:
            analysis['recommendations'].append({
                'plant_id': plant_id,
                'type': 'CAPACITY_INCREASE',
                'priority': 'HIGH',
                'message': f"Plant {plant_id} is over-utilized ({metrics['avg_utilization']:.1f}%). Consider capacity expansion."
            })
        elif metrics['avg_utilization'] < 60:
            analysis['recommendations'].append({
                'plant_id': plant_id,
                'type': 'CAPACITY_OPTIMIZATION',
                'priority': 'MEDIUM',
                'message': f"Plant {plant_id} is under-utilized ({metrics['avg_utilization']:.1f}%). Consider capacity reallocation."
            })
        
        if metrics['avg_fulfillment'] < 95:
            analysis['recommendations'].append({
                'plant_id': plant_id,
                'type': 'FULFILLMENT_IMPROVEMENT',
                'priority': 'HIGH',
                'message': f"Plant {plant_id} has low fulfillment rate ({metrics['avg_fulfillment']:.1f}%). Investigate demand-supply mismatch."
            })
    
    return analysis

def optimize_multi_plant_profit(multi_plant_data, multi_plant_forecasts, plant_capacities, production_costs, plant_pricing):
    """
    Optimize profit across multiple plants considering pricing differences and production costs
    """
    multi_plant_profit_results = {}
    
    for plant_id in multi_plant_forecasts.keys():
        try:
            forecast_df = multi_plant_forecasts[plant_id]
            plant_capacity = plant_capacities.get(plant_id, 1000)
            plant_costs = production_costs.get(plant_id, {'cost_per_unit': 0.60, 'fixed_daily_cost': 100})
            pricing_data = plant_pricing.get(plant_id, {}).get('price_data', pd.DataFrame())
            
            # Create supply dataframe with constant capacity for this plant
            supply_df = pd.DataFrame({
                'ds': forecast_df['ds'],
                'milk_supply': [plant_capacity] * len(forecast_df)
            })
            
            # Optimize for profit
            result_df = optimize_for_profit(forecast_df, pricing_data, plant_costs, plant_capacity, supply_df)
            multi_plant_profit_results[plant_id] = result_df
            
        except Exception as e:
            print(f"Error optimizing profit for plant {plant_id}: {str(e)}")
            continue
    
    return multi_plant_profit_results

def calculate_profit_metrics(multi_plant_profit_results, plant_capacities):
    """
    Calculate comprehensive profit and business metrics for each plant
    """
    profit_metrics = {}
    
    for plant_id, result_df in multi_plant_profit_results.items():
        if result_df is None or result_df.empty:
            continue
            
        try:
            plant_capacity = plant_capacities.get(plant_id, 1000)
            
            # Financial metrics
            total_revenue = result_df['revenue'].sum()
            total_variable_cost = result_df['variable_cost'].sum()
            total_profit = result_df['daily_profit'].sum()
            avg_profit_margin = result_df['profit_margin'].mean()
            
            # Operational metrics
            avg_capacity_utilization = result_df['capacity_utilization'].mean()
            avg_demand_fulfillment = result_df['demand_fulfillment'].mean()
            total_production = result_df['optimal_production'].sum()
            total_sales = result_df['actual_sales'].sum()
            
            # Pricing metrics
            avg_sell_price = result_df['sell_price'].mean()
            price_volatility = result_df['sell_price'].std()
            
            # ROI and efficiency metrics
            roi_percent = (total_profit / total_variable_cost * 100) if total_variable_cost > 0 else 0
            revenue_per_capacity = total_revenue / plant_capacity
            profit_per_capacity = total_profit / plant_capacity
            
            profit_metrics[plant_id] = {
                'plant_id': plant_id,
                'capacity': plant_capacity,
                
                # Financial Performance
                'total_revenue': round(total_revenue, 2),
                'total_variable_cost': round(total_variable_cost, 2),
                'total_profit': round(total_profit, 2),
                'avg_profit_margin': round(avg_profit_margin, 2),
                'roi_percent': round(roi_percent, 2),
                
                # Operational Performance
                'total_production': round(total_production, 2),
                'total_sales': round(total_sales, 2),
                'avg_capacity_utilization': round(avg_capacity_utilization, 2),
                'avg_demand_fulfillment': round(avg_demand_fulfillment, 2),
                
                # Pricing Performance
                'avg_sell_price': round(avg_sell_price, 2),
                'price_volatility': round(price_volatility, 2),
                
                # Efficiency Metrics
                'revenue_per_capacity': round(revenue_per_capacity, 2),
                'profit_per_capacity': round(profit_per_capacity, 2),
                'profit_efficiency_score': calculate_profit_efficiency_score(result_df, plant_capacity)
            }
            
        except Exception as e:
            print(f"Error calculating profit metrics for plant {plant_id}: {str(e)}")
            continue
    
    return profit_metrics

def calculate_profit_efficiency_score(result_df, plant_capacity):
    """
    Calculate a comprehensive profit efficiency score (0-100)
    Combines profit margin, capacity utilization, and demand fulfillment
    """
    if result_df.empty:
        return 0
    
    avg_profit_margin = result_df['profit_margin'].mean()
    avg_utilization = result_df['capacity_utilization'].mean()
    avg_fulfillment = result_df['demand_fulfillment'].mean()
    
    # Normalize profit margin (assume good margin is 20%+)
    normalized_margin = min(100, max(0, avg_profit_margin * 5))  # 20% margin = 100 points
    
    # Weight the components: profit margin (50%), fulfillment (30%), utilization (20%)
    efficiency_score = (
        normalized_margin * 0.5 +
        avg_fulfillment * 0.3 +
        avg_utilization * 0.2
    )
    
    return round(efficiency_score, 2)

def analyze_profit_opportunities(profit_metrics):
    """
    Analyze profit optimization opportunities across plants
    """
    if not profit_metrics:
        return {}
    
    analysis = {
        'summary': {},
        'recommendations': [],
        'performance_ranking': {}
    }
    
    # Summary statistics
    all_profits = [m['total_profit'] for m in profit_metrics.values()]
    all_margins = [m['avg_profit_margin'] for m in profit_metrics.values()]
    all_roi = [m['roi_percent'] for m in profit_metrics.values()]
    
    analysis['summary'] = {
        'total_profit_all_plants': round(sum(all_profits), 2),
        'avg_profit_margin': round(np.mean(all_margins), 2),
        'avg_roi': round(np.mean(all_roi), 2),
        'best_performer': max(profit_metrics.items(), key=lambda x: x[1]['profit_efficiency_score'])[0],
        'highest_profit': max(profit_metrics.items(), key=lambda x: x[1]['total_profit'])[0],
        'highest_margin': max(profit_metrics.items(), key=lambda x: x[1]['avg_profit_margin'])[0]
    }
    
    # Performance ranking
    ranked_plants = sorted(profit_metrics.items(), key=lambda x: x[1]['profit_efficiency_score'], reverse=True)
    for i, (plant_id, metrics) in enumerate(ranked_plants):
        analysis['performance_ranking'][plant_id] = {
            'rank': i + 1,
            'efficiency_score': metrics['profit_efficiency_score'],
            'total_profit': metrics['total_profit'],
            'profit_margin': metrics['avg_profit_margin']
        }
    
    # Generate recommendations
    for plant_id, metrics in profit_metrics.items():
        if metrics['avg_profit_margin'] < 10:
            analysis['recommendations'].append({
                'plant_id': plant_id,
                'priority': 'High',
                'type': 'Profit Margin',
                'message': f"Plant {plant_id} has low profit margin ({metrics['avg_profit_margin']:.1f}%). Consider cost reduction or premium pricing strategies."
            })
        
        if metrics['avg_capacity_utilization'] < 70:
            analysis['recommendations'].append({
                'plant_id': plant_id,
                'priority': 'Medium',
                'type': 'Capacity Utilization',
                'message': f"Plant {plant_id} has low capacity utilization ({metrics['avg_capacity_utilization']:.1f}%). Consider demand stimulation or capacity reallocation."
            })
        
        if metrics['roi_percent'] < 15:
            analysis['recommendations'].append({
                'plant_id': plant_id,
                'priority': 'High',
                'type': 'ROI',
                'message': f"Plant {plant_id} has low ROI ({metrics['roi_percent']:.1f}%). Review operational efficiency and cost structure."
            })
    
    return analysis
