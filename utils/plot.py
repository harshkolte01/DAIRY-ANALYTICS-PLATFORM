import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_forecast(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='ds', y='yhat', data=df, label='Forecasted Demand', linewidth=2)
    if 'yhat_lower' in df.columns and 'yhat_upper' in df.columns:
        plt.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'], alpha=0.3, label='Confidence Interval')
    plt.xticks(rotation=45)
    plt.title("Demand Forecast", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    return plt

def plot_schedule(df):
    plt.figure(figsize=(15, 10))
    
    # Convert date column to datetime if it's string
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Main production vs demand plot
    plt.subplot(2, 2, 1)
    sns.lineplot(x='date', y='optimal_production', data=df, label='Optimal Production', linewidth=2)
    sns.lineplot(x='date', y='forecast_demand', data=df, label='Forecasted Demand', linewidth=2)
    if 'supply_capacity' in df.columns:
        sns.lineplot(x='date', y='supply_capacity', data=df, label='Supply Capacity', linewidth=2, linestyle='--')
    plt.xticks(rotation=45)
    plt.title("Production Schedule vs Demand", fontweight='bold')
    plt.legend()
    
    # Supply utilization plot
    if 'supply_utilization' in df.columns:
        plt.subplot(2, 2, 2)
        sns.lineplot(x='date', y='supply_utilization', data=df, color='green', linewidth=2)
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Max Capacity')
        plt.xticks(rotation=45)
        plt.title("Supply Utilization (%)", fontweight='bold')
        plt.ylabel("Utilization %")
        plt.legend()
    
    # Demand fulfillment plot
    if 'demand_fulfillment' in df.columns:
        plt.subplot(2, 2, 3)
        sns.lineplot(x='date', y='demand_fulfillment', data=df, color='orange', linewidth=2)
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='100% Fulfillment')
        plt.xticks(rotation=45)
        plt.title("Demand Fulfillment (%)", fontweight='bold')
        plt.ylabel("Fulfillment %")
        plt.legend()
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Calculate summary stats
    total_demand = df['forecast_demand'].sum()
    total_production = df['optimal_production'].sum()
    avg_utilization = df['supply_utilization'].mean() if 'supply_utilization' in df.columns else 0
    avg_fulfillment = df['demand_fulfillment'].mean() if 'demand_fulfillment' in df.columns else 0
    
    summary_text = f"""
    SUMMARY STATISTICS
    
    Total Forecasted Demand: {total_demand:,.0f}
    Total Optimal Production: {total_production:,.0f}
    Production vs Demand: {(total_production/total_demand)*100:.1f}%
    
    Average Supply Utilization: {avg_utilization:.1f}%
    Average Demand Fulfillment: {avg_fulfillment:.1f}%
    
    Days with Full Demand Met: {len(df[df['demand_fulfillment'] >= 99.9]) if 'demand_fulfillment' in df.columns else 'N/A'}
    Days with Supply Shortage: {len(df[df['supply_utilization'] >= 99.9]) if 'supply_utilization' in df.columns else 'N/A'}
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    return plt

def plot_supply_analysis(supply_df, calendar_df=None):
    """
    Plot supply analysis showing seasonal and event impacts.
    """
    plt.figure(figsize=(15, 8))
    
    # Main supply trend
    plt.subplot(2, 2, 1)
    sns.lineplot(x='ds', y='milk_supply', data=supply_df, linewidth=2)
    plt.xticks(rotation=45)
    plt.title("Milk Supply Over Time", fontweight='bold')
    plt.ylabel("Daily Supply")
    
    # Seasonal factors
    plt.subplot(2, 2, 2)
    sns.lineplot(x='ds', y='seasonal_factor', data=supply_df, color='green', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
    plt.xticks(rotation=45)
    plt.title("Seasonal Impact on Supply", fontweight='bold')
    plt.ylabel("Seasonal Multiplier")
    plt.legend()
    
    # Event factors
    plt.subplot(2, 2, 3)
    sns.lineplot(x='ds', y='event_factor', data=supply_df, color='orange', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
    plt.xticks(rotation=45)
    plt.title("Event Impact on Supply", fontweight='bold')
    plt.ylabel("Event Multiplier")
    plt.legend()
    
    # Supply distribution by month
    plt.subplot(2, 2, 4)
    supply_df['month'] = supply_df['ds'].dt.month
    monthly_supply = supply_df.groupby('month')['milk_supply'].mean()
    plt.bar(monthly_supply.index, monthly_supply.values, color='skyblue', alpha=0.7)
    plt.title("Average Supply by Month", fontweight='bold')
    plt.xlabel("Month")
    plt.ylabel("Average Daily Supply")
    plt.xticks(range(1, 13))
    
    plt.tight_layout()
    return plt

def plot_multi_plant_comparison(comparative_metrics):
    """
    Create comprehensive multi-plant comparison visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Plant Performance Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data
    plants = list(comparative_metrics.keys())
    efficiencies = [comparative_metrics[p]['efficiency_score'] for p in plants]
    utilizations = [comparative_metrics[p]['avg_utilization'] for p in plants]
    fulfillments = [comparative_metrics[p]['avg_fulfillment'] for p in plants]
    capacities = [comparative_metrics[p]['capacity'] for p in plants]
    demands = [comparative_metrics[p]['total_demand'] for p in plants]
    productions = [comparative_metrics[p]['total_production'] for p in plants]
    
    # 1. Efficiency Score Comparison
    bars1 = axes[0, 0].bar(plants, efficiencies, color='lightblue', alpha=0.8)
    axes[0, 0].set_title('Plant Efficiency Scores', fontweight='bold')
    axes[0, 0].set_ylabel('Efficiency Score (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, eff in zip(bars1, efficiencies):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{eff:.1f}%', ha='center', va='bottom')
    
    # 2. Utilization vs Fulfillment Scatter
    scatter = axes[0, 1].scatter(utilizations, fulfillments, s=[c/10 for c in capacities], 
                                alpha=0.7, c=efficiencies, cmap='viridis')
    axes[0, 1].set_xlabel('Supply Utilization (%)')
    axes[0, 1].set_ylabel('Demand Fulfillment (%)')
    axes[0, 1].set_title('Utilization vs Fulfillment\n(Size = Capacity, Color = Efficiency)', fontweight='bold')
    
    # Add plant labels
    for i, plant in enumerate(plants):
        axes[0, 1].annotate(plant, (utilizations[i], fulfillments[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add colorbar
    plt.colorbar(scatter, ax=axes[0, 1], label='Efficiency Score')
    
    # 3. Capacity vs Demand
    x_pos = np.arange(len(plants))
    width = 0.35
    
    bars2 = axes[0, 2].bar(x_pos - width/2, capacities, width, label='Capacity', alpha=0.8, color='lightcoral')
    bars3 = axes[0, 2].bar(x_pos + width/2, demands, width, label='Demand', alpha=0.8, color='lightgreen')
    
    axes[0, 2].set_title('Capacity vs Demand by Plant', fontweight='bold')
    axes[0, 2].set_ylabel('Volume (Liters)')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(plants, rotation=45)
    axes[0, 2].legend()
    
    # 4. Production Efficiency (Production/Capacity ratio)
    prod_efficiency = [p/c * 100 for p, c in zip(productions, capacities)]
    bars4 = axes[1, 0].bar(plants, prod_efficiency, color='gold', alpha=0.8)
    axes[1, 0].set_title('Production Efficiency\n(Production/Capacity %)', fontweight='bold')
    axes[1, 0].set_ylabel('Production Efficiency (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, eff in zip(bars4, prod_efficiency):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{eff:.1f}%', ha='center', va='bottom')
    
    # 5. Shortage Days Comparison
    shortage_days = [comparative_metrics[p]['shortage_days'] for p in plants]
    bars5 = axes[1, 1].bar(plants, shortage_days, color='salmon', alpha=0.8)
    axes[1, 1].set_title('Shortage Days by Plant', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Shortage Days')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, days in zip(bars5, shortage_days):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(days)}', ha='center', va='bottom')
    
    # 6. Overall Performance Radar Chart (for top 3 plants)
    top_3_plants = sorted(plants, key=lambda p: comparative_metrics[p]['efficiency_score'], reverse=True)[:3]
    
    # Radar chart setup
    categories = ['Efficiency', 'Utilization', 'Fulfillment', 'Capacity Usage']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    ax_radar = plt.subplot(2, 3, 6, projection='polar')
    
    colors = ['blue', 'red', 'green']
    for i, plant in enumerate(top_3_plants):
        metrics = comparative_metrics[plant]
        values = [
            metrics['efficiency_score'],
            metrics['avg_utilization'],
            metrics['avg_fulfillment'],
            (metrics['total_production'] / metrics['capacity']) * 100
        ]
        values += values[:1]  # Complete the circle
        
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=plant, color=colors[i])
        ax_radar.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_title('Top 3 Plants Performance Radar', fontweight='bold', y=1.08)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    return fig

def plot_regional_analysis(regional_insights):
    """
    Create regional performance analysis visualizations
    """
    if not regional_insights:
        # Return empty plot if no regional data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No regional data available\n(requires multiple plants per region)', 
                ha='center', va='center', fontsize=14)
        ax.set_title('Regional Analysis', fontweight='bold')
        return fig
    
    regions = list(regional_insights.keys())
    num_regions = len(regions)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Regional Performance Analysis', fontsize=16, fontweight='bold')
    
    # Regional capacity comparison
    regional_capacities = [regional_insights[r]['total_capacity'] for r in regions]
    regional_demands = [regional_insights[r]['total_demand'] for r in regions]
    
    x_pos = np.arange(len(regions))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, regional_capacities, width, label='Total Capacity', alpha=0.8)
    axes[0, 0].bar(x_pos + width/2, regional_demands, width, label='Total Demand', alpha=0.8)
    axes[0, 0].set_title('Regional Capacity vs Demand')
    axes[0, 0].set_ylabel('Volume (Liters)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(regions)
    axes[0, 0].legend()
    
    # Regional efficiency scores
    regional_efficiencies = [regional_insights[r]['avg_regional_efficiency'] for r in regions]
    bars = axes[0, 1].bar(regions, regional_efficiencies, color='lightblue', alpha=0.8)
    axes[0, 1].set_title('Average Regional Efficiency')
    axes[0, 1].set_ylabel('Efficiency Score (%)')
    
    # Add value labels
    for bar, eff in zip(bars, regional_efficiencies):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{eff:.1f}%', ha='center', va='bottom')
    
    # Number of plants per region
    plants_per_region = [regional_insights[r]['num_plants'] for r in regions]
    axes[1, 0].bar(regions, plants_per_region, color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('Number of Plants per Region')
    axes[1, 0].set_ylabel('Number of Plants')
    
    # Capacity distribution within regions (pie chart for first region)
    if regions:
        first_region = regions[0]
        capacity_dist = regional_insights[first_region]['capacity_distribution']
        
        axes[1, 1].pie(capacity_dist.values(), labels=capacity_dist.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title(f'Capacity Distribution in {first_region}')
    
    plt.tight_layout()
    return fig

def plot_demand_comparison_by_plant(multi_plant_forecasts):
    """
    Compare demand patterns across different plants
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Plant Demand Comparison', fontsize=16, fontweight='bold')
    
    plant_ids = list(multi_plant_forecasts.keys())
    
    # Helper function to get the right demand column
    def get_demand_column(df):
        if 'y' in df.columns:
            return 'y'
        elif 'sales' in df.columns:
            return 'sales'
        else:
            return 'yhat'
    
    # 1. Demand time series overlay
    ax1 = axes[0, 0]
    for plant_id, forecast_df in multi_plant_forecasts.items():
        # Take last 30 days for better visibility
        recent_data = forecast_df.tail(30)
        demand_col = get_demand_column(recent_data)
        ax1.plot(recent_data['ds'], recent_data[demand_col], label=plant_id, linewidth=2)
    
    ax1.set_title('Demand Patterns (Last 30 Days)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Demand')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Average demand comparison
    avg_demands = []
    for plant_id, forecast_df in multi_plant_forecasts.items():
        demand_col = get_demand_column(forecast_df)
        avg_demand = forecast_df[demand_col].mean()
        avg_demands.append(avg_demand)
    
    bars = axes[0, 1].bar(plant_ids, avg_demands, color='skyblue', alpha=0.8)
    axes[0, 1].set_title('Average Daily Demand by Plant')
    axes[0, 1].set_ylabel('Average Daily Demand')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, demand in zip(bars, avg_demands):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(avg_demands)*0.01,
                       f'{demand:.0f}', ha='center', va='bottom')
    
    # 3. Demand variability (coefficient of variation)
    cv_values = []
    for plant_id, forecast_df in multi_plant_forecasts.items():
        demand_col = get_demand_column(forecast_df)
        cv = (forecast_df[demand_col].std() / forecast_df[demand_col].mean()) * 100
        cv_values.append(cv)
    
    axes[1, 0].bar(plant_ids, cv_values, color='lightcoral', alpha=0.8)
    axes[1, 0].set_title('Demand Variability (Coefficient of Variation)')
    axes[1, 0].set_ylabel('CV (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Demand distribution box plot
    demand_data = []
    labels = []
    for plant_id, forecast_df in multi_plant_forecasts.items():
        demand_col = get_demand_column(forecast_df)
        demand_data.append(forecast_df[demand_col].values)
        labels.append(plant_id)
    
    axes[1, 1].boxplot(demand_data, labels=labels)
    axes[1, 1].set_title('Demand Distribution by Plant')
    axes[1, 1].set_ylabel('Daily Demand')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_profit_analysis(profit_metrics):
    """
    Create comprehensive profit analysis visualizations
    """
    if not profit_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No profit data available', ha='center', va='center', fontsize=14)
        ax.set_title('Profit Analysis', fontweight='bold')
        return fig
    
    plant_ids = list(profit_metrics.keys())
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Multi-Plant Profit Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Total Profit by Plant
    total_profits = [profit_metrics[p]['total_profit'] for p in plant_ids]
    bars1 = axes[0, 0].bar(plant_ids, total_profits, color='lightgreen', alpha=0.8)
    axes[0, 0].set_title('Total Profit by Plant', fontweight='bold')
    axes[0, 0].set_ylabel('Total Profit ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, profit in zip(bars1, total_profits):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(total_profits)*0.01,
                       f'${profit:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Profit Margin vs ROI Scatter
    margins = [profit_metrics[p]['avg_profit_margin'] for p in plant_ids]
    roi_values = [profit_metrics[p]['roi_percent'] for p in plant_ids]
    revenues = [profit_metrics[p]['total_revenue'] for p in plant_ids]
    
    scatter = axes[0, 1].scatter(margins, roi_values, s=[r/100 for r in revenues], 
                                alpha=0.7, c=total_profits, cmap='viridis')
    axes[0, 1].set_xlabel('Profit Margin (%)')
    axes[0, 1].set_ylabel('ROI (%)')
    axes[0, 1].set_title('Profit Margin vs ROI\n(Size = Revenue, Color = Profit)', fontweight='bold')
    
    # Add plant labels
    for i, plant in enumerate(plant_ids):
        axes[0, 1].annotate(plant, (margins[i], roi_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, ax=axes[0, 1], label='Total Profit ($)')
    
    # 3. Revenue vs Costs
    revenues = [profit_metrics[p]['total_revenue'] for p in plant_ids]
    costs = [profit_metrics[p]['total_variable_cost'] for p in plant_ids]
    
    x_pos = np.arange(len(plant_ids))
    width = 0.35
    
    bars2 = axes[0, 2].bar(x_pos - width/2, revenues, width, label='Revenue', alpha=0.8, color='skyblue')
    bars3 = axes[0, 2].bar(x_pos + width/2, costs, width, label='Variable Costs', alpha=0.8, color='lightcoral')
    
    axes[0, 2].set_title('Revenue vs Variable Costs', fontweight='bold')
    axes[0, 2].set_ylabel('Amount ($)')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(plant_ids, rotation=45)
    axes[0, 2].legend()
    
    # 4. Profit Efficiency Scores
    efficiency_scores = [profit_metrics[p]['profit_efficiency_score'] for p in plant_ids]
    bars4 = axes[1, 0].bar(plant_ids, efficiency_scores, color='gold', alpha=0.8)
    axes[1, 0].set_title('Profit Efficiency Scores', fontweight='bold')
    axes[1, 0].set_ylabel('Efficiency Score (0-100)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars4, efficiency_scores):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{score:.1f}', ha='center', va='bottom')
    
    # 5. Price vs Utilization Analysis
    avg_prices = [profit_metrics[p]['avg_sell_price'] for p in plant_ids]
    utilizations = [profit_metrics[p]['avg_capacity_utilization'] for p in plant_ids]
    
    axes[1, 1].scatter(avg_prices, utilizations, s=100, alpha=0.7, c='purple')
    axes[1, 1].set_xlabel('Average Sell Price ($)')
    axes[1, 1].set_ylabel('Capacity Utilization (%)')
    axes[1, 1].set_title('Price vs Utilization Strategy', fontweight='bold')
    
    # Add plant labels
    for i, plant in enumerate(plant_ids):
        axes[1, 1].annotate(plant, (avg_prices[i], utilizations[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 6. Performance Ranking
    ranked_plants = sorted(plant_ids, key=lambda p: profit_metrics[p]['profit_efficiency_score'], reverse=True)
    ranks = list(range(1, len(ranked_plants) + 1))
    ranked_scores = [profit_metrics[p]['profit_efficiency_score'] for p in ranked_plants]
    
    bars6 = axes[1, 2].barh(ranked_plants, ranked_scores, color='lightblue', alpha=0.8)
    axes[1, 2].set_title('Plant Performance Ranking', fontweight='bold')
    axes[1, 2].set_xlabel('Efficiency Score')
    
    # Add rank numbers
    for i, (plant, score) in enumerate(zip(ranked_plants, ranked_scores)):
        axes[1, 2].text(score + 1, i, f'#{i+1}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_profit_trends(multi_plant_profit_results):
    """
    Plot profit trends over time for multiple plants
    """
    if not multi_plant_profit_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No profit trend data available', ha='center', va='center', fontsize=14)
        ax.set_title('Profit Trends', fontweight='bold')
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Plant Profit Trends Analysis', fontsize=16, fontweight='bold')
    
    plant_ids = list(multi_plant_profit_results.keys())
    
    # 1. Daily Profit Trends
    ax1 = axes[0, 0]
    for plant_id, result_df in multi_plant_profit_results.items():
        if result_df is not None and not result_df.empty:
            # Take last 30 days for better visibility
            recent_data = result_df.tail(30)
            ax1.plot(recent_data['ds'], recent_data['daily_profit'], label=plant_id, linewidth=2)
    
    ax1.set_title('Daily Profit Trends (Last 30 Days)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Profit ($)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Cumulative Profit
    ax2 = axes[0, 1]
    for plant_id, result_df in multi_plant_profit_results.items():
        if result_df is not None and not result_df.empty:
            cumulative_profit = result_df['daily_profit'].cumsum()
            ax2.plot(result_df['ds'], cumulative_profit, label=plant_id, linewidth=2)
    
    ax2.set_title('Cumulative Profit Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Profit ($)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Profit Margin Trends
    ax3 = axes[1, 0]
    for plant_id, result_df in multi_plant_profit_results.items():
        if result_df is not None and not result_df.empty:
            # Rolling average profit margin
            rolling_margin = result_df['profit_margin'].rolling(window=7, min_periods=1).mean()
            ax3.plot(result_df['ds'], rolling_margin, label=plant_id, linewidth=2)
    
    ax3.set_title('7-Day Rolling Average Profit Margin')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Profit Margin (%)')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Revenue vs Cost Trends
    ax4 = axes[1, 1]
    for plant_id, result_df in multi_plant_profit_results.items():
        if result_df is not None and not result_df.empty:
            # Calculate weekly totals for better visibility
            result_df['week'] = result_df['ds'].dt.isocalendar().week
            weekly_data = result_df.groupby('week').agg({
                'revenue': 'sum',
                'variable_cost': 'sum'
            }).reset_index()
            
            ax4.plot(weekly_data['week'], weekly_data['revenue'], 
                    label=f'{plant_id} Revenue', linewidth=2, linestyle='-')
            ax4.plot(weekly_data['week'], weekly_data['variable_cost'], 
                    label=f'{plant_id} Costs', linewidth=2, linestyle='--')
    
    ax4.set_title('Weekly Revenue vs Costs')
    ax4.set_xlabel('Week of Year')
    ax4.set_ylabel('Amount ($)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig
