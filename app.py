import streamlit as st
import pandas as pd
import numpy as np
import time
from utils.data_loader import (
    load_sales_data, load_calendar_data, preprocess_sales_data, 
    simulate_milk_supply, get_available_stores_and_states,
    preprocess_sales_data_by_plant, get_plant_capacity_mapping,
    load_prices_data, get_item_pricing_data, calculate_average_price_by_plant,
    simulate_production_costs
)
from utils.ml_data_loader import (
    MLEnhancedDataLoader, get_ml_model_summary, get_advanced_analytics_summary
)
from utils.feature_engineering import create_comprehensive_features
from utils.forecast import forecast_demand
from utils.optimizer import (
    optimize_schedule, analyze_supply_demand_gaps, get_optimization_summary,
    optimize_multi_plant_schedule, compare_plant_performance,
    optimize_for_profit, optimize_multi_plant_profit, calculate_profit_metrics,
    analyze_profit_opportunities
)
from utils.plot import (
    plot_forecast, plot_schedule, plot_supply_analysis,
    plot_multi_plant_comparison, plot_regional_analysis, 
    plot_demand_comparison_by_plant, plot_profit_analysis, plot_profit_trends
)

st.set_page_config(page_title="Dairy Analytics Platform", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'forecast_df' not in st.session_state:
    st.session_state['forecast_df'] = None
if 'supply_df' not in st.session_state:
    st.session_state['supply_df'] = None
if 'ml_loader' not in st.session_state:
    st.session_state['ml_loader'] = None
if 'ml_insights' not in st.session_state:
    st.session_state['ml_insights'] = None
if 'pretrained_models_available' not in st.session_state:
    # Check if pre-trained models exist
    import os
    pretrained_path = 'models/latest_trained_models.pkl'
    st.session_state['pretrained_models_available'] = os.path.exists(pretrained_path)

# Load data once and cache in session state
if 'data_loaded' not in st.session_state:
    st.session_state['sales_df'] = load_sales_data()
    st.session_state['calendar_df'] = load_calendar_data()
    st.session_state['ts_data'] = preprocess_sales_data(st.session_state['sales_df'], st.session_state['calendar_df'])
    st.session_state['data_loaded'] = True

# Sidebar Navigation
st.sidebar.title("🥛 Dairy Analytics Platform")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["🏠 Home - Demand Forecasting", "🚚 Supply Simulation", "🔄 Optimization & Analysis", "🏭 Multi-Plant Analysis", "💰 Cost & Profit Optimization", "🤖 Advanced ML Analytics", "📊 Dashboard & Reports"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This platform helps optimize dairy operations by combining demand forecasting, "
    "supply simulation, and production optimization with real-world constraints."
)

# HOME PAGE - DEMAND FORECASTING
if page == "🏠 Home - Demand Forecasting":
    st.title("🏠 Demand Forecasting Dashboard")
    st.markdown("Welcome to the Dairy Analytics Platform! Start by generating demand forecasts based on historical sales data.")
    
    # Data Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sales Records", f"{len(st.session_state['sales_df']):,}")
    
    with col2:
        st.metric("Calendar Days", f"{len(st.session_state['calendar_df']):,}")
    
    with col3:
        st.metric("Time Series Length", f"{len(st.session_state['ts_data'])} days")
    
    st.markdown("---")
    
    # Data Preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Historical Sales Data Preview")
        st.dataframe(st.session_state['ts_data'].tail(10), height=300)
        
        # Basic statistics
        st.subheader("📊 Sales Statistics")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.write(f"**Average Daily Sales:** {st.session_state['ts_data']['sales'].mean():.1f}")
            st.write(f"**Maximum Sales:** {st.session_state['ts_data']['sales'].max():.1f}")
        with stats_col2:
            st.write(f"**Minimum Sales:** {st.session_state['ts_data']['sales'].min():.1f}")
            st.write(f"**Total Sales:** {st.session_state['ts_data']['sales'].sum():,.0f}")
    
    with col2:
        st.subheader("🔮 Demand Forecasting")
        st.markdown("Generate demand forecasts using Facebook Prophet algorithm that considers trends, seasonality, and holidays.")
        
        if st.button("🚀 Run Demand Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating forecast... This may take a moment."):
                forecast_df = forecast_demand(st.session_state['ts_data'])
                st.session_state['forecast_df'] = forecast_df
                st.success("✅ Forecast completed successfully!")
        
        if st.session_state['forecast_df'] is not None:
            st.subheader("📋 Forecast Results")
            st.dataframe(st.session_state['forecast_df'].tail(10), height=200)
            
            forecast_stats_col1, forecast_stats_col2 = st.columns(2)
            with forecast_stats_col1:
                st.write(f"**Avg Forecast:** {st.session_state['forecast_df']['yhat'].mean():.1f}")
                st.write(f"**Max Forecast:** {st.session_state['forecast_df']['yhat'].max():.1f}")
            with forecast_stats_col2:
                st.write(f"**Min Forecast:** {st.session_state['forecast_df']['yhat'].min():.1f}")
                st.write(f"**Forecast Period:** {len(st.session_state['forecast_df'])} days")
    
    # Forecast Visualization
    if st.session_state['forecast_df'] is not None:
        st.markdown("---")
        st.subheader("📈 Forecast Visualization")
        st.pyplot(plot_forecast(st.session_state['forecast_df']))
        
        st.info("💡 **Next Step:** Navigate to 'Supply Simulation' to model milk supply constraints based on seasonal and event factors.")

# SUPPLY SIMULATION PAGE
elif page == "🚚 Supply Simulation":
    st.title("🚚 Milk Supply Simulation")
    st.markdown("Configure and simulate milk supply considering seasonal trends, calendar events, and operational constraints.")
    
    if st.session_state['forecast_df'] is None:
        st.warning("⚠️ Please generate demand forecasts first from the Home page.")
        st.stop()
    
    # Configuration Section
    st.subheader("⚙️ Supply Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("**🏭 Base Capacity**")
        base_supply = st.slider("Daily Base Supply", 500, 2000, 800, 50)
        st.caption("Base daily milk supply capacity in liters")
    
    with config_col2:
        st.markdown("**🌡️ Seasonal Impact**")
        seasonal_factor = st.slider("Seasonal Variation", 0.0, 0.5, 0.2, 0.05)
        st.caption("How much seasons affect supply (0=none, 0.5=high)")
    
    with config_col3:
        st.markdown("**🎉 Event Impact**")
        event_factor = st.slider("Event Impact Factor", 0.0, 0.3, 0.15, 0.05)
        st.caption("How much events affect supply logistics")
    
    st.markdown("---")
    
    # Supply Simulation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Supply Modeling Factors")
        st.markdown("""
        **Seasonal Effects:**
        - 🌨️ **Winter (Dec-Feb):** Higher supply (+10%)
        - 🌸 **Spring (Mar-May):** Moderate supply
        - ☀️ **Summer (Jun-Aug):** Lower supply (-25% in July)
        - 🍂 **Fall (Sep-Nov):** Recovering supply
        
        **Event-Based Impacts:**
        - 🎄 **Major Holidays:** -80% logistics efficiency
        - 🎭 **Cultural Events:** -50% logistics efficiency  
        - 🏛️ **National Holidays:** -60% logistics efficiency
        - 🏈 **Sporting Events:** -20% logistics efficiency
        - 📅 **Weekends:** -5% operations
        """)
    
    with col2:
        st.subheader("▶️ Run Supply Simulation")
        
        if st.button("🔄 Generate Supply Simulation", type="primary", use_container_width=True):
            with st.spinner("Simulating milk supply patterns..."):
                supply_df = simulate_milk_supply(
                    st.session_state['calendar_df'], 
                    base_supply=base_supply,
                    seasonal_factor=seasonal_factor,
                    event_factor=event_factor
                )
                st.session_state['supply_df'] = supply_df
                st.success("✅ Supply simulation completed!")
        
        if st.session_state['supply_df'] is not None:
            st.markdown("**📊 Supply Statistics:**")
            supply_stats = st.session_state['supply_df']['milk_supply']
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Average Supply", f"{supply_stats.mean():.0f}L")
                st.metric("Minimum Supply", f"{supply_stats.min():.0f}L")
            with metrics_col2:
                st.metric("Maximum Supply", f"{supply_stats.max():.0f}L")
                st.metric("Supply Variability", f"{supply_stats.std():.0f}L")
    
    # Supply Data and Visualization
    if st.session_state['supply_df'] is not None:
        st.markdown("---")
        
        # Supply data preview
        st.subheader("📋 Supply Data Preview")
        st.dataframe(st.session_state['supply_df'].tail(15), height=300)
        
        # Supply analysis charts
        st.subheader("📈 Supply Analysis Charts")
        st.pyplot(plot_supply_analysis(st.session_state['supply_df'], st.session_state['calendar_df']))
        
        st.info("💡 **Next Step:** Navigate to 'Optimization & Analysis' to compare supply with demand and optimize production schedules.")

# OPTIMIZATION & ANALYSIS PAGE
elif page == "🔄 Optimization & Analysis":
    st.title("🔄 Production Optimization & Gap Analysis")
    st.markdown("Analyze supply-demand gaps and optimize production schedules under real constraints.")
    
    if st.session_state['forecast_df'] is None or st.session_state['supply_df'] is None:
        st.warning("⚠️ Please complete demand forecasting and supply simulation first.")
        st.stop()
    
    # Quick Overview
    col1, col2, col3, col4 = st.columns(4)
    
    avg_demand = st.session_state['forecast_df']['yhat'].mean()
    avg_supply = st.session_state['supply_df']['milk_supply'].mean()
    supply_demand_ratio = (avg_supply / avg_demand) * 100
    
    with col1:
        st.metric("Avg Daily Demand", f"{avg_demand:.0f}L")
    with col2:
        st.metric("Avg Daily Supply", f"{avg_supply:.0f}L")
    with col3:
        st.metric("Supply/Demand Ratio", f"{supply_demand_ratio:.1f}%")
    with col4:
        if supply_demand_ratio >= 100:
            st.success("✅ Supply Adequate")
        else:
            st.error("⚠️ Supply Shortage")
    
    st.markdown("---")
    
    # Analysis Options
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.subheader("📊 Supply-Demand Gap Analysis")
        st.markdown("Identify periods of supply surplus or shortage.")
        
        if st.button("🔍 Analyze Supply-Demand Gaps", type="primary", use_container_width=True):
            with st.spinner("Analyzing supply-demand gaps..."):
                gap_df, summary = analyze_supply_demand_gaps(
                    st.session_state['forecast_df'], 
                    st.session_state['supply_df']
                )
                st.session_state['gap_df'] = gap_df
                st.session_state['gap_summary'] = summary
                st.success("✅ Gap analysis completed!")
        
        if 'gap_summary' in st.session_state:
            st.markdown("**📈 Gap Analysis Summary:**")
            summary = st.session_state['gap_summary']
            
            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.metric("Surplus Days", f"{summary['surplus_days']}")
                st.metric("Avg Surplus", f"{summary['avg_surplus']:.0f}L")
            with summary_col2:
                st.metric("Shortage Days", f"{summary['shortage_days']}")
                st.metric("Avg Shortage", f"{summary['avg_shortage']:.0f}L")
            
            st.dataframe(st.session_state['gap_df'].tail(10), height=200)
    
    with analysis_col2:
        st.subheader("⚙️ Production Optimization")
        st.markdown("Find optimal production schedule balancing supply constraints and demand.")
        
        if st.button("🎯 Optimize Production Schedule", type="primary", use_container_width=True):
            with st.spinner("Optimizing production schedule..."):
                result_df = optimize_schedule(
                    st.session_state['forecast_df'], 
                    st.session_state['supply_df']
                )
                st.session_state['optimization_result'] = result_df
                st.success("✅ Optimization completed!")
        
        if 'optimization_result' in st.session_state:
            result_df = st.session_state['optimization_result']
            
            # Get optimization summary
            summary = get_optimization_summary(result_df)
            
            st.markdown("**🎯 Optimization Results:**")
            
            opt_col1, opt_col2, opt_col3 = st.columns(3)
            with opt_col1:
                st.metric("Avg Supply Utilization", f"{summary['avg_utilization']:.1f}%")
            with opt_col2:
                st.metric("Avg Demand Fulfillment", f"{summary['avg_fulfillment']:.1f}%")
            with opt_col3:
                status_color = "🟢" if summary['status'] == "Success" else "🔴"
                st.metric("Optimization Status", f"{status_color} {summary['status']}")
            
            # Additional metrics
            st.markdown("**🔍 Detailed Analysis:**")
            detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
            
            with detail_col1:
                st.metric("Total Production", f"{summary['total_production']:,.0f}L")
            with detail_col2:
                st.metric("Total Demand", f"{summary['total_demand']:,.0f}L")
            with detail_col3:
                st.metric("Shortage Days", summary['shortage_days'])
            with detail_col4:
                st.metric("Low Utilization Days", summary['surplus_days'])
            
            st.dataframe(result_df.tail(10), height=200)
    
    # Optimization Visualization
    if 'optimization_result' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Optimization Visualization")
        st.pyplot(plot_schedule(st.session_state['optimization_result']))

# MULTI-PLANT ANALYSIS PAGE
elif page == "🏭 Multi-Plant Analysis":
    st.title("🏭 Multi-Plant Analysis Dashboard")
    st.markdown("Compare demand and utilization across multiple store_ids representing different dairy plants or regions.")
    
    # Load available stores and states
    stores_states = get_available_stores_and_states(st.session_state['sales_df'])
    
    if not stores_states:
        st.error("❌ No store/state data found in the dataset.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏪 Available Plants (Store IDs)")
        selected_stores = st.multiselect(
            "Select plants to analyze (minimum 2):",
            options=list(stores_states.keys()),
            default=list(stores_states.keys())[:5] if len(stores_states) >= 5 else list(stores_states.keys()),
            help="Each store_id represents a different dairy plant or facility."
        )
    
    with col2:
        st.subheader("🗺️ Plant Locations")
        if selected_stores:
            # Create a simple dataframe for display
            plant_data = []
            for store in selected_stores:
                plant_data.append({
                    'Plant_ID': store,
                    'State': stores_states[store]
                })
            plant_df = pd.DataFrame(plant_data)
            st.dataframe(plant_df, height=200, use_container_width=True)
    
    if len(selected_stores) < 2:
        st.warning("⚠️ Please select at least 2 plants for comparison analysis.")
        st.stop()
    
    st.markdown("---")
    
    # Multi-Plant Analysis Controls
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Comprehensive Multi-Plant Comparison", "Regional Performance Analysis", "Demand Pattern Comparison"]
    )
    
    if st.button("🚀 Run Multi-Plant Analysis", type="primary", use_container_width=True):
        
        with st.spinner("Processing multi-plant analysis... This may take a moment."):
            
            # Load and preprocess data for selected plants
            multi_plant_data = {}
            multi_plant_forecasts = {}
            
            for store_id in selected_stores:
                plant_data = preprocess_sales_data_by_plant(
                    st.session_state['sales_df'], 
                    st.session_state['calendar_df'], 
                    store_id
                )
                if not plant_data.empty:
                    multi_plant_data[store_id] = plant_data
                    # Generate forecast for each plant
                    forecast_df = forecast_demand(plant_data)
                    multi_plant_forecasts[store_id] = forecast_df
            
            if not multi_plant_data:
                st.error("❌ No valid data found for selected plants.")
                st.stop()
            
            # Get plant capacity mapping
            capacity_mapping = get_plant_capacity_mapping(selected_stores, stores_states)
            
            if analysis_type == "Comprehensive Multi-Plant Comparison":
                st.subheader("🏭 Comprehensive Multi-Plant Comparison")
                
                # Run multi-plant optimization
                try:
                    optimization_results = optimize_multi_plant_schedule(multi_plant_data, capacity_mapping)
                    
                    if optimization_results:
                        # Analyze performance
                        comparative_metrics = compare_plant_performance(
                            multi_plant_data, capacity_mapping, optimization_results
                        )
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Performance Metrics")
                            metrics_df = pd.DataFrame(comparative_metrics).T
                            st.dataframe(metrics_df.round(2), height=300)
                        
                        with col2:
                            st.subheader("🎯 Key Performance Indicators")
                            best_plant = max(comparative_metrics.keys(), 
                                           key=lambda x: comparative_metrics[x]['efficiency_score'])
                            worst_plant = min(comparative_metrics.keys(), 
                                            key=lambda x: comparative_metrics[x]['efficiency_score'])
                            
                            st.metric("Best Performing Plant", best_plant, 
                                     f"{comparative_metrics[best_plant]['efficiency_score']:.1f}%")
                            st.metric("Lowest Performing Plant", worst_plant, 
                                     f"{comparative_metrics[worst_plant]['efficiency_score']:.1f}%")
                            
                            avg_efficiency = np.mean([m['efficiency_score'] for m in comparative_metrics.values()])
                            st.metric("Average Plant Efficiency", f"{avg_efficiency:.1f}%")
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("📈 Multi-Plant Performance Visualization")
                        fig = plot_multi_plant_comparison(comparative_metrics)
                        st.pyplot(fig)
                        
                    else:
                        st.error("❌ Multi-plant optimization failed. Please check your data.")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
            
            elif analysis_type == "Regional Performance Analysis":
                st.subheader("🗺️ Regional Performance Analysis")
                
                # Group plants by region (state)
                regional_data = {}
                for store_id in selected_stores:
                    state = stores_states[store_id]
                    if state not in regional_data:
                        regional_data[state] = []
                    if store_id in multi_plant_data:
                        regional_data[state].append({
                            'store_id': store_id,
                            'data': multi_plant_data[store_id],
                            'capacity': capacity_mapping.get(store_id, 1000)
                        })
                
                # Regional insights
                regional_insights = {}
                for region, plants in regional_data.items():
                    if len(plants) > 0:
                        total_capacity = sum(p['capacity'] for p in plants)
                        total_demand = sum(p['data']['sales'].sum() for p in plants)
                        avg_efficiency = np.mean([
                            min(100, (p['data']['sales'].sum() / p['capacity']) * 100) 
                            for p in plants
                        ])
                        
                        regional_insights[region] = {
                            'num_plants': len(plants),
                            'total_capacity': total_capacity,
                            'total_demand': total_demand,
                            'avg_regional_efficiency': avg_efficiency,
                            'capacity_distribution': {
                                p['store_id']: p['capacity'] for p in plants
                            }
                        }
                
                if regional_insights:
                    # Display regional metrics
                    st.subheader("📋 Regional Metrics Summary")
                    regional_df = pd.DataFrame(regional_insights).T
                    st.dataframe(regional_df, height=200)
                    
                    # Regional visualization
                    st.markdown("---")
                    st.subheader("📊 Regional Analysis Visualization")
                    fig = plot_regional_analysis(regional_insights)
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Insufficient data for regional analysis.")
            
            elif analysis_type == "Demand Pattern Comparison":
                st.subheader("📈 Demand Pattern Comparison")
                
                if multi_plant_forecasts:
                    # Display demand statistics
                    demand_stats = {}
                    for plant_id, forecast_df in multi_plant_forecasts.items():
                        # Use 'y' column which contains historical + forecasted values
                        # If 'y' doesn't exist, fall back to 'sales' or 'yhat'
                        if 'y' in forecast_df.columns:
                            demand_col = 'y'
                        elif 'sales' in forecast_df.columns:
                            demand_col = 'sales'
                        else:
                            demand_col = 'yhat'
                        
                        demand_stats[plant_id] = {
                            'avg_demand': forecast_df[demand_col].mean(),
                            'max_demand': forecast_df[demand_col].max(),
                            'min_demand': forecast_df[demand_col].min(),
                            'demand_volatility': forecast_df[demand_col].std(),
                            'total_demand': forecast_df[demand_col].sum()
                        }
                    
                    st.subheader("📊 Demand Statistics by Plant")
                    demand_df = pd.DataFrame(demand_stats).T
                    st.dataframe(demand_df.round(2), height=300)
                    
                    # Demand pattern visualization
                    st.markdown("---")
                    st.subheader("📈 Demand Pattern Visualization")
                    fig = plot_demand_comparison_by_plant(multi_plant_forecasts)
                    st.pyplot(fig)
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Plant Performance Recommendations")
                    
                    # Find plants with highest and lowest demand
                    avg_demands = {pid: stats['avg_demand'] for pid, stats in demand_stats.items()}
                    highest_demand_plant = max(avg_demands.keys(), key=lambda x: avg_demands[x])
                    lowest_demand_plant = min(avg_demands.keys(), key=lambda x: avg_demands[x])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"🏆 **High Demand Plant:** {highest_demand_plant}")
                        st.write(f"Average daily demand: {avg_demands[highest_demand_plant]:.0f}")
                        st.write("Consider expanding capacity or optimizing distribution.")
                    
                    with col2:
                        st.info(f"📉 **Low Demand Plant:** {lowest_demand_plant}")
                        st.write(f"Average daily demand: {avg_demands[lowest_demand_plant]:.0f}")
                        st.write("Explore opportunities for market expansion or reallocation.")
                
                else:
                    st.error("❌ No forecast data available for demand pattern analysis.")
    
    # Additional Information
    st.markdown("---")
    st.subheader("ℹ️ Multi-Plant Analysis Information")
    st.info(
        "**Multi-Plant Analysis Features:**\n\n"
        "• **Comprehensive Comparison:** Compare efficiency, utilization, and performance across all selected plants\n"
        "• **Regional Analysis:** Group plants by geographic regions (states) for territorial insights\n"
        "• **Demand Patterns:** Analyze demand variability and forecasting accuracy across plants\n"
        "• **Capacity Planning:** Evaluate plant capacities against actual demand patterns\n"
        "• **Performance Benchmarking:** Identify best and worst performing facilities for improvement opportunities"
    )

# COST & PROFIT OPTIMIZATION PAGE
elif page == "💰 Cost & Profit Optimization":
    st.title("💰 Cost & Profit Optimization Dashboard")
    st.markdown("Optimize for maximum profit using real pricing data, production costs, and ROI analysis.")
    
    # Load pricing data
    if 'prices_df' not in st.session_state:
        with st.spinner("Loading pricing data..."):
            st.session_state['prices_df'] = load_prices_data()
    
    # Profit optimization overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Price Records", f"{len(st.session_state['prices_df']):,}")
    
    with col2:
        avg_price = st.session_state['prices_df']['sell_price'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col3:
        price_range = st.session_state['prices_df']['sell_price'].max() - st.session_state['prices_df']['sell_price'].min()
        st.metric("Price Range", f"${price_range:.2f}")
    
    st.markdown("---")
    
    # Profit optimization type selection
    optimization_type = st.selectbox(
        "Select Profit Optimization Analysis:",
        ["Single Plant Profit Analysis", "Multi-Plant Profit Comparison", "ROI & Cost Analysis"]
    )
    
    if optimization_type == "Single Plant Profit Analysis":
        st.subheader("🏪 Single Plant Profit Optimization")
        
        # Get available stores
        stores_states = get_available_stores_and_states(st.session_state['sales_df'])
        selected_store = st.selectbox("Select Plant (Store ID):", list(stores_states.keys()))
        
        col1, col2 = st.columns(2)
        with col1:
            item_id = st.selectbox("Select Product:", ['FOODS_3_090', 'FOODS_3_091', 'FOODS_3_092'])
        with col2:
            base_cost = st.slider("Base Production Cost per Unit ($):", 0.40, 1.00, 0.60, 0.05)
        
        if st.button("🚀 Run Single Plant Profit Analysis", type="primary"):
            with st.spinner("Analyzing profit optimization..."):
                
                # Process data for selected plant
                plant_data = preprocess_sales_data_by_plant(
                    st.session_state['sales_df'], 
                    st.session_state['calendar_df'], 
                    selected_store, 
                    item_id
                )
                
                if plant_data.empty:
                    st.error("❌ No data found for selected plant and item.")
                    st.stop()
                
                # Generate forecast
                forecast_df = forecast_demand(plant_data)
                
                # Get pricing data
                pricing_data = get_item_pricing_data(
                    st.session_state['prices_df'], 
                    st.session_state['calendar_df'], 
                    item_id, 
                    selected_store
                )
                
                # Simulate production costs
                capacity_mapping = get_plant_capacity_mapping([selected_store], stores_states)
                plant_capacity = capacity_mapping[selected_store]
                production_costs = {
                    'cost_per_unit': base_cost,
                    'fixed_daily_cost': plant_capacity * 0.1
                }
                
                # Run profit optimization
                profit_result = optimize_for_profit(
                    forecast_df, pricing_data, production_costs, plant_capacity
                )
                
                if profit_result is not None:
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Profit Metrics")
                        total_profit = profit_result['daily_profit'].sum()
                        total_revenue = profit_result['revenue'].sum()
                        avg_margin = profit_result['profit_margin'].mean()
                        
                        st.metric("Total Profit", f"${total_profit:,.2f}")
                        st.metric("Total Revenue", f"${total_revenue:,.2f}")
                        st.metric("Average Margin", f"{avg_margin:.1f}%")
                    
                    with col2:
                        st.subheader("📈 Performance Indicators")
                        avg_utilization = profit_result['capacity_utilization'].mean()
                        avg_fulfillment = profit_result['demand_fulfillment'].mean()
                        
                        st.metric("Capacity Utilization", f"{avg_utilization:.1f}%")
                        st.metric("Demand Fulfillment", f"{avg_fulfillment:.1f}%")
                    
                    # Detailed results table
                    st.markdown("---")
                    st.subheader("📋 Detailed Profit Analysis")
                    st.dataframe(profit_result.round(2), height=300)
                    
                    # Profit trends chart
                    st.markdown("---")
                    st.subheader("📈 Profit Trends")
                    
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    fig.suptitle(f'Profit Analysis for Plant {selected_store}', fontsize=16)
                    
                    # Daily profit
                    axes[0, 0].plot(profit_result['ds'], profit_result['daily_profit'])
                    axes[0, 0].set_title('Daily Profit')
                    axes[0, 0].set_ylabel('Profit ($)')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    
                    # Revenue vs costs
                    axes[0, 1].plot(profit_result['ds'], profit_result['revenue'], label='Revenue')
                    axes[0, 1].plot(profit_result['ds'], profit_result['variable_cost'], label='Variable Cost')
                    axes[0, 1].set_title('Revenue vs Variable Costs')
                    axes[0, 1].set_ylabel('Amount ($)')
                    axes[0, 1].legend()
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    
                    # Profit margin
                    axes[1, 0].plot(profit_result['ds'], profit_result['profit_margin'])
                    axes[1, 0].set_title('Profit Margin %')
                    axes[1, 0].set_ylabel('Margin (%)')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    
                    # Price trends
                    axes[1, 1].plot(profit_result['ds'], profit_result['sell_price'])
                    axes[1, 1].set_title('Selling Price')
                    axes[1, 1].set_ylabel('Price ($)')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                else:
                    st.error("❌ Profit optimization failed. Please check your parameters.")
    
    elif optimization_type == "Multi-Plant Profit Comparison":
        st.subheader("🏭 Multi-Plant Profit Comparison")
        
        # Get available stores
        stores_states = get_available_stores_and_states(st.session_state['sales_df'])
        selected_stores = st.multiselect(
            "Select Plants for Profit Comparison:",
            options=list(stores_states.keys()),
            default=list(stores_states.keys())[:4]
        )
        
        if len(selected_stores) < 2:
            st.warning("⚠️ Please select at least 2 plants for comparison.")
            st.stop()
        
        col1, col2 = st.columns(2)
        with col1:
            item_id = st.selectbox("Select Product:", ['FOODS_3_090', 'FOODS_3_091', 'FOODS_3_092'], key="multi_item")
        with col2:
            base_cost = st.slider("Base Production Cost per Unit ($):", 0.40, 1.00, 0.60, 0.05, key="multi_cost")
        
        if st.button("🚀 Run Multi-Plant Profit Analysis", type="primary"):
            with st.spinner("Analyzing multi-plant profit optimization..."):
                
                # Process data for all selected plants
                multi_plant_data = {}
                multi_plant_forecasts = {}
                
                for store_id in selected_stores:
                    plant_data = preprocess_sales_data_by_plant(
                        st.session_state['sales_df'], 
                        st.session_state['calendar_df'], 
                        store_id, 
                        item_id
                    )
                    if not plant_data.empty:
                        multi_plant_data[store_id] = plant_data
                        forecast_df = forecast_demand(plant_data)
                        multi_plant_forecasts[store_id] = forecast_df
                
                if not multi_plant_data:
                    st.error("❌ No valid data found for selected plants.")
                    st.stop()
                
                # Get capacity and cost data
                capacity_mapping = get_plant_capacity_mapping(selected_stores, stores_states)
                production_costs = simulate_production_costs(capacity_mapping, base_cost)
                
                # Get pricing data for all plants
                plant_pricing = calculate_average_price_by_plant(
                    st.session_state['prices_df'], 
                    st.session_state['calendar_df'], 
                    item_id, 
                    selected_stores
                )
                
                # Run multi-plant profit optimization
                profit_results = optimize_multi_plant_profit(
                    multi_plant_data, multi_plant_forecasts, capacity_mapping, production_costs, plant_pricing
                )
                
                if profit_results:
                    # Calculate profit metrics
                    profit_metrics = calculate_profit_metrics(profit_results, capacity_mapping)
                    
                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    
                    total_profit = sum([m['total_profit'] for m in profit_metrics.values()])
                    avg_margin = np.mean([m['avg_profit_margin'] for m in profit_metrics.values()])
                    best_plant = max(profit_metrics.items(), key=lambda x: x[1]['profit_efficiency_score'])[0]
                    
                    with col1:
                        st.metric("Total Profit (All Plants)", f"${total_profit:,.2f}")
                    with col2:
                        st.metric("Average Profit Margin", f"{avg_margin:.1f}%")
                    with col3:
                        st.metric("Best Performing Plant", best_plant)
                    
                    # Detailed metrics table
                    st.markdown("---")
                    st.subheader("📊 Plant Performance Comparison")
                    metrics_df = pd.DataFrame(profit_metrics).T
                    st.dataframe(metrics_df.round(2), height=400)
                    
                    # Profit analysis visualization
                    st.markdown("---")
                    st.subheader("📈 Profit Analysis Visualization")
                    fig1 = plot_profit_analysis(profit_metrics)
                    st.pyplot(fig1)
                    
                    # Profit trends visualization
                    st.markdown("---")
                    st.subheader("📈 Profit Trends Over Time")
                    fig2 = plot_profit_trends(profit_results)
                    st.pyplot(fig2)
                    
                    # Profit optimization recommendations
                    st.markdown("---")
                    st.subheader("💡 Profit Optimization Recommendations")
                    recommendations = analyze_profit_opportunities(profit_metrics)
                    
                    if recommendations.get('recommendations'):
                        for rec in recommendations['recommendations']:
                            if rec['priority'] == 'High':
                                st.error(f"🔴 **{rec['type']}** - {rec['message']}")
                            elif rec['priority'] == 'Medium':
                                st.warning(f"🟡 **{rec['type']}** - {rec['message']}")
                            else:
                                st.info(f"🔵 **{rec['type']}** - {rec['message']}")
                    else:
                        st.success("✅ All plants are performing within acceptable profit parameters!")
                
                else:
                    st.error("❌ Multi-plant profit optimization failed.")
    
    elif optimization_type == "ROI & Cost Analysis":
        st.subheader("📊 ROI & Cost Structure Analysis")
        
        st.info(
            "**ROI & Cost Analysis provides:**\n\n"
            "• **Return on Investment (ROI)** calculations for each plant\n"
            "• **Cost structure breakdown** (variable vs fixed costs)\n"
            "• **Price sensitivity analysis** showing impact of pricing changes\n"
            "• **Economies of scale analysis** comparing plant sizes and costs\n"
            "• **Investment recommendations** for capacity expansion or optimization"
        )
        
        # Cost parameters input
        st.subheader("🔧 Cost Structure Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            base_variable_cost = st.slider("Base Variable Cost per Unit ($):", 0.30, 1.00, 0.60, 0.05)
        with col2:
            fixed_cost_multiplier = st.slider("Fixed Cost Multiplier:", 0.05, 0.20, 0.10, 0.01)
        with col3:
            target_roi = st.slider("Target ROI (%):", 10, 50, 20, 5)
        
        if st.button("🚀 Run ROI & Cost Analysis", type="primary"):
            with st.spinner("Analyzing ROI and cost structures..."):
                
                # Get sample of stores for analysis
                stores_states = get_available_stores_and_states(st.session_state['sales_df'])
                sample_stores = list(stores_states.keys())[:6]  # Analyze 6 plants
                
                # Generate cost analysis data
                capacity_mapping = get_plant_capacity_mapping(sample_stores, stores_states)
                production_costs = simulate_production_costs(capacity_mapping, base_variable_cost)
                
                # Calculate ROI metrics for each plant
                roi_analysis = {}
                for store_id in sample_stores:
                    capacity = capacity_mapping[store_id]
                    var_cost = production_costs[store_id]['cost_per_unit']
                    fixed_cost = production_costs[store_id]['fixed_daily_cost']
                    
                    # Simulate annual metrics
                    annual_capacity = capacity * 365
                    annual_fixed_cost = fixed_cost * 365
                    
                    # Assume average utilization and pricing
                    avg_utilization = np.random.uniform(0.70, 0.95)
                    avg_price = st.session_state['prices_df']['sell_price'].mean()
                    
                    annual_production = annual_capacity * avg_utilization
                    annual_revenue = annual_production * avg_price
                    annual_variable_cost = annual_production * var_cost
                    annual_profit = annual_revenue - annual_variable_cost - annual_fixed_cost
                    
                    # Calculate ROI (profit / total investment)
                    total_investment = annual_fixed_cost + (annual_variable_cost * 0.1)  # Assume 10% working capital
                    roi_percent = (annual_profit / total_investment * 100) if total_investment > 0 else 0
                    
                    roi_analysis[store_id] = {
                        'capacity': capacity,
                        'annual_capacity': annual_capacity,
                        'utilization': avg_utilization * 100,
                        'annual_revenue': annual_revenue,
                        'annual_variable_cost': annual_variable_cost,
                        'annual_fixed_cost': annual_fixed_cost,
                        'annual_profit': annual_profit,
                        'roi_percent': roi_percent,
                        'meets_target': roi_percent >= target_roi,
                        'variable_cost_per_unit': var_cost,
                        'profit_margin': (annual_profit / annual_revenue * 100) if annual_revenue > 0 else 0
                    }
                
                # Display ROI analysis results
                st.subheader("📊 ROI Analysis Results")
                roi_df = pd.DataFrame(roi_analysis).T
                st.dataframe(roi_df.round(2), height=300)
                
                # ROI visualization
                st.markdown("---")
                st.subheader("📈 ROI & Cost Analysis Visualization")
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle('ROI & Cost Structure Analysis', fontsize=16, fontweight='bold')
                
                plant_ids = list(roi_analysis.keys())
                
                # 1. ROI by plant
                roi_values = [roi_analysis[p]['roi_percent'] for p in plant_ids]
                colors = ['green' if roi >= target_roi else 'red' for roi in roi_values]
                bars1 = axes[0, 0].bar(plant_ids, roi_values, color=colors, alpha=0.7)
                axes[0, 0].axhline(y=target_roi, color='orange', linestyle='--', label=f'Target ROI ({target_roi}%)')
                axes[0, 0].set_title('ROI by Plant')
                axes[0, 0].set_ylabel('ROI (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].legend()
                
                # 2. Cost structure breakdown
                fixed_costs = [roi_analysis[p]['annual_fixed_cost'] for p in plant_ids]
                variable_costs = [roi_analysis[p]['annual_variable_cost'] for p in plant_ids]
                
                x_pos = np.arange(len(plant_ids))
                width = 0.35
                axes[0, 1].bar(x_pos - width/2, fixed_costs, width, label='Fixed Costs', alpha=0.8)
                axes[0, 1].bar(x_pos + width/2, variable_costs, width, label='Variable Costs', alpha=0.8)
                axes[0, 1].set_title('Annual Cost Structure')
                axes[0, 1].set_ylabel('Cost ($)')
                axes[0, 1].set_xticks(x_pos)
                axes[0, 1].set_xticklabels(plant_ids, rotation=45)
                axes[0, 1].legend()
                
                # 3. Profit margin vs capacity
                margins = [roi_analysis[p]['profit_margin'] for p in plant_ids]
                capacities = [roi_analysis[p]['capacity'] for p in plant_ids]
                scatter = axes[0, 2].scatter(capacities, margins, s=100, alpha=0.7, c=roi_values, cmap='RdYlGn')
                axes[0, 2].set_xlabel('Plant Capacity')
                axes[0, 2].set_ylabel('Profit Margin (%)')
                axes[0, 2].set_title('Capacity vs Profit Margin')
                plt.colorbar(scatter, ax=axes[0, 2], label='ROI (%)')
                
                # 4. Revenue breakdown
                revenues = [roi_analysis[p]['annual_revenue'] for p in plant_ids]
                profits = [roi_analysis[p]['annual_profit'] for p in plant_ids]
                axes[1, 0].bar(plant_ids, revenues, alpha=0.7, label='Revenue')
                axes[1, 0].bar(plant_ids, profits, alpha=0.7, label='Profit')
                axes[1, 0].set_title('Annual Revenue vs Profit')
                axes[1, 0].set_ylabel('Amount ($)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].legend()
                
                # 5. Cost per unit analysis
                cost_per_unit = [roi_analysis[p]['variable_cost_per_unit'] for p in plant_ids]
                utilizations = [roi_analysis[p]['utilization'] for p in plant_ids]
                axes[1, 1].scatter(cost_per_unit, utilizations, s=100, alpha=0.7)
                axes[1, 1].set_xlabel('Variable Cost per Unit ($)')
                axes[1, 1].set_ylabel('Capacity Utilization (%)')
                axes[1, 1].set_title('Cost vs Utilization Efficiency')
                
                # Add plant labels
                for i, plant in enumerate(plant_ids):
                    axes[1, 1].annotate(plant, (cost_per_unit[i], utilizations[i]))
                
                # 6. Investment efficiency
                investment_efficiency = [roi_analysis[p]['annual_profit'] / roi_analysis[p]['capacity'] for p in plant_ids]
                bars6 = axes[1, 2].bar(plant_ids, investment_efficiency, color='purple', alpha=0.7)
                axes[1, 2].set_title('Profit per Unit Capacity')
                axes[1, 2].set_ylabel('Profit per Capacity Unit ($)')
                axes[1, 2].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Investment recommendations
                st.markdown("---")
                st.subheader("💡 Investment Recommendations")
                
                meeting_target = [p for p in plant_ids if roi_analysis[p]['meets_target']]
                not_meeting_target = [p for p in plant_ids if not roi_analysis[p]['meets_target']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"✅ **Plants Meeting Target ROI ({target_roi}%):**")
                    for plant in meeting_target:
                        roi = roi_analysis[plant]['roi_percent']
                        st.write(f"• {plant}: {roi:.1f}% ROI")
                
                with col2:
                    if not_meeting_target:
                        st.error(f"❌ **Plants Below Target ROI:**")
                        for plant in not_meeting_target:
                            roi = roi_analysis[plant]['roi_percent']
                            st.write(f"• {plant}: {roi:.1f}% ROI")
                    else:
                        st.success("🎉 All plants meet the target ROI!")
    
    # Additional Information
    st.markdown("---")
    st.subheader("ℹ️ Cost & Profit Optimization Information")
    st.info(
        "**Cost & Profit Optimization Features:**\n\n"
        "• **Revenue Optimization:** Uses real M5 pricing data (sell_prices.csv) for accurate revenue calculations\n"
        "• **Cost Modeling:** Simulates variable and fixed production costs based on plant capacity and efficiency\n"
        "• **Profit Maximization:** Linear programming optimization to maximize profit instead of just capacity utilization\n"
        "• **ROI Analysis:** Comprehensive return on investment calculations for business decision making\n"
        "• **Multi-Plant Comparison:** Compare profitability across different plants and regions\n"
        "• **Investment Guidance:** Data-driven recommendations for capacity expansion and operational improvements"
    )

# ADVANCED ML ANALYTICS PAGE
elif page == "🤖 Advanced ML Analytics":
    st.title("🤖 Advanced Machine Learning Analytics")
    st.markdown("Leverage advanced ML models trained on M5 dataset for deeper business insights and predictive analytics.")
    
    # ML Overview
    st.subheader("🧠 Available ML Models")
    ml_summary = get_ml_model_summary()
    
    # Check for pre-trained models
    if st.session_state['pretrained_models_available']:
        st.success("✅ Pre-trained models detected! You can use them for instant analysis or train new ones.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Pre-trained Models", type="primary"):
                with st.spinner("Loading pre-trained models..."):
                    if st.session_state['ml_loader'] is None:
                        st.session_state['ml_loader'] = MLEnhancedDataLoader()
                    
                    success = st.session_state['ml_loader'].load_models('models/latest_trained_models.pkl')
                    if success:
                        st.session_state['ml_insights'] = st.session_state['ml_loader'].get_business_insights()
                        st.success("✅ Pre-trained models loaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load pre-trained models")
        
        with col2:
            if st.button("🔄 Train New Models", type="secondary"):
                st.session_state['ml_loader'] = None  # Reset to force new training
                st.info("Scroll down to 'Full ML Model Training' section to train new models")
    else:
        st.info("💡 No pre-trained models found. You can train models interactively or run the pre-training script.")
        st.code("python train_models.py --full_train", language="bash")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Predictive Models:**")
        for model_name, model_info in ml_summary['models'].items():
            with st.expander(f"📈 {model_name.replace('_', ' ').title()}"):
                st.write(f"**Purpose:** {model_info['purpose']}")
                st.write(f"**Input:** {model_info['input']}")
                st.write(f"**Output:** {model_info['output']}")
                st.write(f"**Business Value:** {model_info['business_value']}")
    
        with col2:
            st.markdown("**🔧 Feature Engineering:**")
            for feature_type, description in ml_summary['features'].items():
                st.write(f"• **{feature_type.replace('_', ' ').title()}:** {description}")
    
    st.markdown("---")
    
    # Model Status Display
    st.subheader("📊 Model Status")
    
    if st.session_state['ml_loader'] and st.session_state['ml_loader'].is_trained:
        st.success("✅ ML Models are loaded and ready!")
        
        # Display loaded model info
        trained_models = st.session_state['ml_loader'].trained_models
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Loaded", len(trained_models))
        with col2:
            insights_count = len(st.session_state.get('ml_insights', {}))
            st.metric("Insights Available", insights_count)
        with col3:
            st.metric("Status", "✅ Ready")
        
        # Quick insights preview
        if st.session_state.get('ml_insights'):
            with st.expander("🔍 Quick Insights Preview"):
                insights = st.session_state['ml_insights']
                
                if 'feature_insights' in insights:
                    feature_insights = insights['feature_insights']
                    
                    if 'seasonality' in feature_insights:
                        seasonality = feature_insights['seasonality']
                        st.write(f"**Peak Months:** {seasonality.get('peak_months', [])}")
                        st.write(f"**Low Months:** {seasonality.get('low_months', [])}")
                    
                    if 'event_impact' in feature_insights:
                        event_count = len(feature_insights['event_impact'])
                        st.write(f"**Events Analyzed:** {event_count}")
                
                if 'recommendations' in insights:
                    rec_count = len(insights['recommendations'])
                    st.write(f"**Recommendations Generated:** {rec_count}")
    else:
        st.warning("⚠️ No ML models loaded. Choose an option above to get started.")
    
    st.markdown("---")    # ML Analysis Options
    analysis_option = st.selectbox(
        "Select ML Analysis Type:",
        [
            "Quick Analytics Summary", 
            "Advanced Feature Analysis", 
            "Full ML Model Training",
            "Store Performance ML Analysis"
        ]
    )
    
    if analysis_option == "Quick Analytics Summary":
        st.subheader("⚡ Quick ML Analytics Summary")
        st.markdown("Get instant insights without training complex models.")
        
        if st.button("🚀 Generate Quick Analytics", type="primary"):
            with st.spinner("Generating ML-powered analytics..."):
                
                # Get advanced analytics summary
                analytics_summary = get_advanced_analytics_summary(
                    st.session_state['sales_df'],
                    st.session_state['calendar_df'],
                    st.session_state.get('prices_df')
                )
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ML Features Created", analytics_summary['feature_count'])
                
                with col2:
                    anomaly_pct = analytics_summary.get('anomaly_percentage', 0)
                    st.metric("Data Anomalies", f"{anomaly_pct:.1f}%")
                
                with col3:
                    recommendations_count = len(analytics_summary.get('recommendations', []))
                    st.metric("ML Recommendations", recommendations_count)
                
                with col4:
                    insights_count = len(analytics_summary.get('insights', {}))
                    st.metric("Business Insights", insights_count)
                
                # Display insights
                st.markdown("---")
                st.subheader("🔍 Key Business Insights")
                
                insights = analytics_summary.get('insights', {})
                
                if 'price_sensitivity' in insights:
                    st.markdown("**💰 Price Sensitivity Analysis:**")
                    price_data = insights['price_sensitivity']
                    for feature, sensitivity in price_data.items():
                        if abs(sensitivity) > 0.1:
                            sensitivity_level = "High" if abs(sensitivity) > 0.3 else "Medium"
                            direction = "Negative" if sensitivity < 0 else "Positive"
                            st.write(f"• {feature}: {direction} correlation ({sensitivity:.2f}) - {sensitivity_level} sensitivity")
                
                if 'event_impact' in insights:
                    st.markdown("**🎉 Event Impact Analysis:**")
                    for event, impact_data in insights['event_impact'].items():
                        impact_pct = impact_data.get('impact_percentage', 0)
                        if abs(impact_pct) > 5:
                            direction = "increases" if impact_pct > 0 else "decreases"
                            st.write(f"• {event} {direction} demand by {abs(impact_pct):.1f}%")
                
                if 'seasonality' in insights:
                    st.markdown("**📅 Seasonal Patterns:**")
                    seasonality = insights['seasonality']
                    st.write(f"• Peak months: {seasonality.get('peak_months', [])}")
                    st.write(f"• Low months: {seasonality.get('low_months', [])}")
                
                # Display recommendations
                st.markdown("---")
                st.subheader("💡 ML-Generated Recommendations")
                
                recommendations = analytics_summary.get('recommendations', [])
                for i, rec in enumerate(recommendations):
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get('priority', 'low'), "🔵")
                    st.write(f"{priority_color} **{rec.get('type', 'General').title()}:** {rec.get('message', '')}")
                    if 'action' in rec:
                        st.write(f"   → *Action:* {rec['action']}")
    
    elif analysis_option == "Advanced Feature Analysis":
        st.subheader("🔬 Advanced Feature Engineering Analysis")
        st.markdown("Deep dive into feature creation and importance analysis.")
        
        # Store selection for detailed analysis
        stores_states = get_available_stores_and_states(st.session_state['sales_df'])
        selected_store = st.selectbox("Select Store for Detailed Analysis:", list(stores_states.keys())[:10])
        
        if st.button("🔍 Analyze Features", type="primary"):
            with st.spinner("Creating and analyzing advanced features..."):
                
                # Filter data for selected store
                store_sales = st.session_state['sales_df'][
                    st.session_state['sales_df']['store_id'] == selected_store
                ]
                
                if len(store_sales) == 0:
                    st.error("No data found for selected store.")
                    st.stop()
                
                # Create comprehensive features
                feature_data = create_comprehensive_features(
                    store_sales,
                    st.session_state['calendar_df'],
                    st.session_state.get('prices_df')
                )
                
                # Display feature statistics
                features_df = feature_data['features']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Features", len(features_df.columns))
                with col2:
                    st.metric("Data Points", len(features_df))
                with col3:
                    st.metric("Feature Density", f"{(features_df != 0).mean().mean():.1%}")
                
                # Feature categories
                st.markdown("---")
                st.subheader("📊 Feature Categories")
                
                feature_categories = {}
                for col in features_df.columns:
                    category = 'other'
                    if 'time' in col or 'day' in col or 'month' in col or 'week' in col:
                        category = 'time_features'
                    elif 'lag' in col:
                        category = 'lag_features'
                    elif 'mean' in col or 'std' in col or 'min' in col or 'max' in col:
                        category = 'rolling_features'
                    elif 'price' in col:
                        category = 'price_features'
                    elif 'snap' in col or 'event' in col:
                        category = 'event_features'
                    elif 'sin' in col or 'cos' in col:
                        category = 'cyclical_features'
                    
                    if category not in feature_categories:
                        feature_categories[category] = 0
                    feature_categories[category] += 1
                
                # Display feature category breakdown
                category_df = pd.DataFrame(
                    list(feature_categories.items()),
                    columns=['Category', 'Count']
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(category_df, height=200)
                
                with col2:
                    # Simple bar chart of feature categories
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(category_df['Category'], category_df['Count'])
                    ax.set_title('Feature Categories Distribution')
                    ax.set_xlabel('Category')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Feature correlation with target
                st.markdown("---")
                st.subheader("🎯 Feature Correlation with Sales")
                
                # Calculate correlation with original sales data
                store_ts = preprocess_sales_data(store_sales, st.session_state['calendar_df'])
                
                if len(store_ts) == len(features_df):
                    correlations = features_df.corrwith(store_ts['sales']).abs().sort_values(ascending=False)
                    top_features = correlations.head(15)
                    
                    # Display top correlated features
                    fig, ax = plt.subplots(figsize=(10, 8))
                    top_features.plot(kind='barh', ax=ax)
                    ax.set_title('Top 15 Features by Correlation with Sales')
                    ax.set_xlabel('Absolute Correlation')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.dataframe(
                        pd.DataFrame({
                            'Feature': top_features.index,
                            'Correlation': top_features.values
                        }).round(3),
                        height=300
                    )
                
                # Display insights and recommendations
                st.markdown("---")
                st.subheader("💡 Feature Analysis Insights")
                
                insights = feature_data['insights']
                recommendations = feature_data['recommendations']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔍 Key Insights:**")
                    for insight_type, insight_data in insights.items():
                        st.write(f"• **{insight_type.replace('_', ' ').title()}:**")
                        if isinstance(insight_data, dict):
                            for key, value in list(insight_data.items())[:3]:
                                st.write(f"  - {key}: {value}")
                
                with col2:
                    st.markdown("**🎯 Recommendations:**")
                    for rec in recommendations[:5]:
                        priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get('priority', 'low'), "🔵")
                        st.write(f"{priority_color} {rec.get('message', '')}")
    
    elif analysis_option == "Full ML Model Training":
        st.subheader("🎯 Complete ML Model Training with Progress Tracking")
        st.markdown("Train all ML models for comprehensive predictive analytics with real-time progress monitoring.")
        
        # Training options
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("⚠️ **Interactive Training**")
            st.write("• Real-time progress tracking")
            st.write("• Estimated time: 5-8 minutes")
            st.write("• Live feedback and ETA")
            st.write("• Full feature engineering")
        
        with col2:
            st.info("💡 **Pro Tip**: For command line training, use:")
            st.code("python train_models.py", language="bash")
            st.write("• Full training by default (~8 minutes)")
            st.write("• Pre-saves models for instant loading")
            st.write("• Add --quick_train for faster testing (~3 minutes)")
        
        # Get available stores
        stores_states = get_available_stores_and_states(st.session_state['sales_df'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_store = st.selectbox("Select Store for Training:", ["All Stores"] + list(stores_states.keys())[:5])
            train_item = st.selectbox("Select Item for Training:", ["All Items", "FOODS_3_090", "FOODS_3_091", "FOODS_3_092"])
        
        with col2:
            st.markdown("**🤖 Models to be trained:**")
            st.write("• 🎯 Demand Spike Classifier (RF)")
            st.write("• 📈 XGBoost Volume Regressor")
            st.write("• ⚡ LightGBM Volume Regressor")
            st.write("• 🌲 Random Forest Regressor")
            st.write("• 📅 Seasonality Analyzer")
            st.markdown("**📊 Features:** 100+ engineered features")
        
        # Training button with enhanced warning
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Start Interactive ML Training", type="primary", use_container_width=True):
                
                # Enhanced training with progress tracking
                training_start_time = time.time()
                
                # Create progress containers
                progress_container = st.container()
                with progress_container:
                    st.markdown("### 🔄 Training Progress")
                    
                    # Main progress bar
                    main_progress = st.progress(0)
                    status_text = st.empty()
                    timing_text = st.empty()
                    
                    # Step-by-step progress
                    step_container = st.container()
                
                # Initialize ML loader
                if st.session_state['ml_loader'] is None:
                    st.session_state['ml_loader'] = MLEnhancedDataLoader()
                
                ml_loader = st.session_state['ml_loader']
                
                # Prepare training parameters
                store_id = None if train_store == "All Stores" else train_store
                item_id = None if train_item == "All Items" else train_item
                
                try:
                    # Enhanced progress callback
                    def streamlit_progress_callback(step_progress, message):
                        elapsed = time.time() - training_start_time
                        
                        # Update main progress
                        main_progress.progress(step_progress)
                        status_text.text(f"🔄 {message}")
                        
                        # Calculate and display timing
                        if step_progress > 0:
                            total_estimated = elapsed / step_progress
                            eta_seconds = total_estimated - elapsed
                            eta = max(0, eta_seconds)
                            timing_text.text(f"⏱️ Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | Progress: {step_progress*100:.1f}%")
                        else:
                            timing_text.text(f"⏱️ Elapsed: {elapsed:.0f}s | Starting...")
                        
                        # Update step container
                        with step_container:
                            if step_progress < 0.2:
                                st.text("📊 Step 1/6: Loading and preparing data...")
                            elif step_progress < 0.35:
                                st.text("🔧 Step 2/6: Creating 100+ advanced features...")
                            elif step_progress < 0.6:
                                st.text("🤖 Step 3/6: Training ML models (this takes the most time)...")
                            elif step_progress < 0.8:
                                st.text("🧠 Step 4/6: Generating ML insights...")
                            elif step_progress < 0.95:
                                st.text("💾 Step 5/6: Saving trained models...")
                            else:
                                st.text("✅ Step 6/6: Training completed!")
                    
                    # Start training with progress tracking
                    status_text.text("🚀 Initializing ML training pipeline...")
                    
                    training_results = ml_loader.train_ml_models(
                        store_id=store_id,
                        item_id=item_id,
                        save_models=True,
                        use_streamlit=True  # Enable Streamlit progress tracking
                    )
                    
                    # Final progress update
                    main_progress.progress(1.0)
                    status_text.text("🎉 Training completed successfully!")
                    
                    total_time = time.time() - training_start_time
                    timing_text.text(f"✅ Total training time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
                    
                    st.session_state['ml_insights'] = training_results['insights']
                    
                    # Success message with celebration
                    st.balloons()
                    st.success(f"🎉 ML models trained successfully in {total_time/60:.1f} minutes!")
                    
                    # Display training results
                    st.markdown("---")
                    st.subheader("📊 Training Results & Model Performance")
                    
                    # Performance metrics overview
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                    
                    total_accuracy = 0
                    total_r2 = 0
                    model_count = 0
                    accuracy_count = 0
                    
                    for model_name, results in training_results['training_results'].items():
                        if 'accuracy' in results:
                            total_accuracy += results['accuracy']
                            accuracy_count += 1
                        if 'test_r2' in results:
                            total_r2 += results['test_r2']
                            model_count += 1
                    
                    with perf_col1:
                        if accuracy_count > 0:
                            avg_accuracy = total_accuracy / accuracy_count
                            st.metric("Avg Classification Accuracy", f"{avg_accuracy:.1%}")
                        else:
                            st.metric("Models Trained", len(training_results['training_results']))
                    
                    with perf_col2:
                        if model_count > 0:
                            avg_r2 = total_r2 / model_count
                            st.metric("Avg Regression R²", f"{avg_r2:.3f}")
                        else:
                            st.metric("Feature Count", "100+")
                    
                    with perf_col3:
                        training_time_min = total_time / 60
                        st.metric("Training Time", f"{training_time_min:.1f} min")
                    
                    with perf_col4:
                        insights_count = len(training_results.get('insights', {}))
                        st.metric("Insights Generated", insights_count)
                    
                    # Detailed model results
                    st.markdown("### 🔍 Detailed Model Performance")
                    
                    for model_name, results in training_results['training_results'].items():
                        with st.expander(f"📈 {model_name.replace('_', ' ').title()} - Detailed Results"):
                            
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                if 'accuracy' in results:
                                    accuracy = results['accuracy']
                                    accuracy_color = "🟢" if accuracy > 0.8 else "🟡" if accuracy > 0.6 else "🔴"
                                    st.metric("Classification Accuracy", f"{accuracy:.1%}", help=f"{accuracy_color} Quality indicator")
                                
                                if 'test_mae' in results:
                                    st.metric("Test MAE (Lower is better)", f"{results['test_mae']:.2f}")
                                
                                if 'test_r2' in results:
                                    r2 = results['test_r2']
                                    r2_color = "🟢" if r2 > 0.8 else "🟡" if r2 > 0.6 else "🔴"
                                    st.metric("Test R² Score", f"{r2:.3f}", help=f"{r2_color} Prediction quality")
                            
                            with result_col2:
                                if 'feature_importance' in results and results['feature_importance'] is not None:
                                    st.markdown("**🔝 Top 5 Important Features:**")
                                    top_features = results['feature_importance'].head(5)
                                    for idx, row in top_features.iterrows():
                                        importance_pct = (row['importance'] / top_features['importance'].max()) * 100
                                        bar_width = int(importance_pct / 5)  # Scale to 20 chars max
                                        bar = "█" * bar_width + "░" * (20 - bar_width)
                                        st.text(f"{row['feature'][:20]}: [{bar}] {row['importance']:.3f}")
                    
                    # Display ML insights
                    st.markdown("---")
                    st.subheader("🧠 AI-Generated Business Insights")
                    
                    insights = st.session_state['ml_insights']
                    
                    if 'demand_spikes' in insights:
                        spike_data = insights['demand_spikes']
                        
                        st.markdown("#### 📈 Demand Prediction Insights")
                        spike_col1, spike_col2, spike_col3 = st.columns(3)
                        
                        with spike_col1:
                            spikes = spike_data.get('upcoming_spikes', 0)
                            spike_color = "🔴" if spikes > 5 else "🟡" if spikes > 2 else "🟢"
                            st.metric("Predicted Demand Spikes", f"{spike_color} {spikes}")
                        
                        with spike_col2:
                            drops = spike_data.get('upcoming_drops', 0)
                            drop_color = "🔴" if drops > 5 else "🟡" if drops > 2 else "🟢"
                            st.metric("Predicted Demand Drops", f"{drop_color} {drops}")
                        
                        with spike_col3:
                            total_predictions = spikes + drops
                            st.metric("Total Anomalies Predicted", total_predictions)
                    
                    if 'feature_insights' in insights:
                        feature_insights = insights['feature_insights']
                        
                        if 'event_impact' in feature_insights:
                            st.markdown("#### 🎉 Event Impact Analysis")
                            event_impacts = []
                            for event, impact in feature_insights['event_impact'].items():
                                impact_pct = impact.get('impact_percentage', 0)
                                if abs(impact_pct) > 2:  # Only show significant impacts
                                    direction = "📈" if impact_pct > 0 else "📉"
                                    event_impacts.append({
                                        'Event': event.replace('_', ' ').title(),
                                        'Impact': f"{direction} {impact_pct:+.1f}%",
                                        'Significance': "High" if abs(impact_pct) > 10 else "Medium" if abs(impact_pct) > 5 else "Low"
                                    })
                            
                            if event_impacts:
                                st.dataframe(pd.DataFrame(event_impacts), use_container_width=True)
                    
                    if 'recommendations' in insights:
                        st.markdown("#### 💡 AI-Powered Business Recommendations")
                        for i, rec in enumerate(insights['recommendations'][:5], 1):
                            priority_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                            priority_color = priority_colors.get(rec.get('priority', 'low'), "🔵")
                            
                            with st.expander(f"{priority_color} Recommendation {i}: {rec.get('type', 'General').title()}"):
                                st.write(rec.get('message', ''))
                                if 'impact' in rec:
                                    st.info(f"💰 Expected Impact: {rec['impact']}")
                                if 'timeline' in rec:
                                    st.info(f"⏰ Timeline: {rec['timeline']}")
                    
                    # Save confirmation
                    st.markdown("---")
                    st.success("💾 **Models Automatically Saved!** You can now use these trained models for instant predictions.")
                    st.info("🚀 **Next Steps:** Use other analysis options above to leverage your trained models for business insights!")
                
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.write("**Troubleshooting tips:**")
                    st.write("• Try selecting a specific store instead of 'All Stores'")
                    st.write("• Ensure sufficient memory (4GB+ recommended)")
                    st.write("• Check data quality and completeness")
                    st.write("• Consider using the command-line training script for better performance")
        
        # Alternative training methods
        st.markdown("---")
        st.markdown("### 🛠️ Alternative Training Methods")
        
        alt_col1, alt_col2 = st.columns(2)
        
        alt_col1, alt_col2 = st.columns(2)
        
        with alt_col1:
            st.markdown("#### ⚡ Quick Command-Line Training")
            st.code("""
# Full training - default (8-10 minutes)
python train_models.py

# Quick training for testing (3 minutes)  
python train_models.py --quick_train
            """, language="bash")
            st.write("• Faster than interactive training")
            st.write("• Progress bar in terminal")
            st.write("• Models saved automatically")
        
        with alt_col2:
            st.markdown("#### 🚀 Batch Training Scripts")
            st.code("""
# Windows
train_models.bat

# Linux/Mac
./train_models.sh
            """, language="bash")
            st.write("• One-click training")
            st.write("• No setup required")
            st.write("• Pre-configured settings")
    
    elif analysis_option == "Store Performance ML Analysis":
        st.subheader("🏪 Multi-Store ML Performance Analysis")
        st.markdown("Compare ML-powered insights across multiple stores.")
        
        # Get available stores
        stores_states = get_available_stores_and_states(st.session_state['sales_df'])
        
        # Store selection
        available_stores = list(stores_states.keys())
        selected_stores = st.multiselect(
            "Select Stores for ML Analysis:",
            available_stores,
            default=available_stores[:3]
        )
        
        if len(selected_stores) < 2:
            st.warning("Please select at least 2 stores for comparison.")
        else:
            if st.button("🔍 Analyze Stores with ML", type="primary"):
                with st.spinner(f"Analyzing {len(selected_stores)} stores with ML..."):
                    
                    # Initialize ML loader
                    if st.session_state['ml_loader'] is None:
                        st.session_state['ml_loader'] = MLEnhancedDataLoader()
                    
                    ml_loader = st.session_state['ml_loader']
                    
                    # Analyze stores
                    store_analysis = ml_loader.analyze_store_performance(selected_stores)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Store Performance Comparison")
                    
                    # Create comparison metrics
                    comparison_data = []
                    for store_id, analysis in store_analysis.items():
                        data_quality = analysis['data_quality']
                        insights = analysis['insights']
                        
                        comparison_data.append({
                            'Store_ID': store_id,
                            'Total_Records': data_quality['total_records'],
                            'Avg_Daily_Sales': data_quality['avg_daily_sales'],
                            'Sales_Volatility': data_quality['sales_volatility'],
                            'Recommendations': len(analysis['recommendations']),
                            'Anomaly_Rate': analysis['anomalies'].get('statistical_outliers', {}).get('percentage', 0)
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df.round(2), height=200)
                    
                    # Individual store insights
                    for store_id, analysis in store_analysis.items():
                        with st.expander(f"🏪 Store {store_id} - Detailed ML Analysis"):
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📈 Performance Metrics:**")
                                data_quality = analysis['data_quality']
                                st.write(f"• Records: {data_quality['total_records']:,}")
                                st.write(f"• Avg Daily Sales: {data_quality['avg_daily_sales']:.1f}")
                                st.write(f"• Volatility: {data_quality['sales_volatility']:.1f}")
                                st.write(f"• Date Range: {data_quality['date_range']}")
                            
                            with col2:
                                st.markdown("**🔍 ML Insights:**")
                                insights = analysis['insights']
                                
                                if 'seasonality' in insights:
                                    seasonality = insights['seasonality']
                                    st.write(f"• Peak Months: {seasonality.get('peak_months', [])}")
                                    st.write(f"• Low Months: {seasonality.get('low_months', [])}")
                                
                                if 'day_of_week_patterns' in insights:
                                    dow_patterns = insights['day_of_week_patterns']
                                    best_day = dow_patterns.get('best_day', 'N/A')
                                    worst_day = dow_patterns.get('worst_day', 'N/A')
                                    st.write(f"• Best Day: {best_day}")
                                    st.write(f"• Worst Day: {worst_day}")
                            
                            # Recommendations
                            st.markdown("**💡 Store-Specific Recommendations:**")
                            for rec in analysis['recommendations'][:3]:
                                priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get('priority', 'low'), "🔵")
                                st.write(f"{priority_color} {rec.get('message', '')}")
    
    # ML Information Section
    st.markdown("---")
    st.subheader("ℹ️ Advanced ML Analytics Information")
    st.info(
        "**ML Analytics Capabilities:**\n\n"
        "• **Demand Spike Prediction:** Classify upcoming demand spikes/drops using Random Forest\n"
        "• **Volume Regression:** Predict exact demand quantities using XGBoost, LightGBM, Random Forest\n"
        "• **Feature Engineering:** Create 100+ advanced features from time series, pricing, and event data\n"
        "• **Seasonality Analysis:** Decompose complex seasonal patterns using Prophet\n"
        "• **Business Insights:** Extract actionable insights from ML models\n"
        "• **Anomaly Detection:** Identify unusual demand patterns and data quality issues\n"
        "• **Cross-Store Analysis:** Compare ML insights across multiple store locations"
    )

# DASHBOARD & REPORTS PAGE
elif page == "📊 Dashboard & Reports":
    st.title("📊 Executive Dashboard & Reports")
    st.markdown("Comprehensive overview of all analytics and key performance indicators.")
    
    if st.session_state['forecast_df'] is None:
        st.warning("⚠️ Please complete the analysis from previous pages first.")
        st.stop()
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    
    exec_col1, exec_col2, exec_col3, exec_col4 = st.columns(4)
    
    with exec_col1:
        st.metric(
            "Forecast Period", 
            f"{len(st.session_state['forecast_df'])} days",
            help="Total number of days forecasted"
        )
    
    with exec_col2:
        if st.session_state['supply_df'] is not None:
            avg_supply = st.session_state['supply_df']['milk_supply'].mean()
            st.metric("Avg Daily Supply", f"{avg_supply:.0f}L")
        else:
            st.metric("Supply Analysis", "Not Available")
    
    with exec_col3:
        avg_demand = st.session_state['forecast_df']['yhat'].mean()
        st.metric("Avg Daily Demand", f"{avg_demand:.0f}L")
    
    with exec_col4:
        if 'optimization_result' in st.session_state:
            avg_fulfillment = st.session_state['optimization_result']['demand_fulfillment'].mean()
            st.metric("Demand Fulfillment", f"{avg_fulfillment:.1f}%")
        else:
            st.metric("Optimization", "Not Available")
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**🎯 Recommendations:**")
        if st.session_state['supply_df'] is not None and 'gap_summary' in st.session_state:
            summary = st.session_state['gap_summary']
            if summary['shortage_days'] > summary['surplus_days']:
                st.error("⚠️ **Action Required:** Consider increasing base supply capacity or improving logistics during events.")
                st.markdown("- Shortage days exceed surplus days")
                st.markdown(f"- Average shortage: {summary['avg_shortage']:.0f}L")
            else:
                st.success("✅ **Good Balance:** Supply generally meets demand with adequate buffer.")
                st.markdown(f"- Surplus days: {summary['surplus_days']}")
                st.markdown(f"- Average surplus: {summary['avg_surplus']:.0f}L")
        else:
            st.info("Complete supply analysis to see recommendations.")
    
    with insights_col2:
        st.markdown("**📈 Performance Metrics:**")
        if 'optimization_result' in st.session_state:
            result_df = st.session_state['optimization_result']
            perfect_days = len(result_df[result_df['demand_fulfillment'] >= 99.9])
            total_days = len(result_df)
            
            st.markdown(f"- **Perfect Fulfillment Days:** {perfect_days}/{total_days}")
            st.markdown(f"- **Avg Supply Utilization:** {result_df['supply_utilization'].mean():.1f}%")
            st.markdown(f"- **Total Production:** {result_df['optimal_production'].sum():,.0f}L")
        else:
            st.info("Run optimization to see performance metrics.")
    
    # Data Export
    st.markdown("---")
    st.subheader("💾 Data Export")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.session_state['forecast_df'] is not None:
            csv_forecast = st.session_state['forecast_df'].to_csv(index=False)
            st.download_button(
                "📥 Download Forecast Data",
                csv_forecast,
                "demand_forecast.csv",
                "text/csv"
            )
    
    with export_col2:
        if st.session_state['supply_df'] is not None:
            csv_supply = st.session_state['supply_df'].to_csv(index=False)
            st.download_button(
                "📥 Download Supply Data",
                csv_supply,
                "supply_simulation.csv",
                "text/csv"
            )
    
    with export_col3:
        if 'optimization_result' in st.session_state:
            csv_optimization = st.session_state['optimization_result'].to_csv(index=False)
            st.download_button(
                "📥 Download Optimization Results",
                csv_optimization,
                "optimization_results.csv",
                "text/csv"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 System Status")
if st.session_state['forecast_df'] is not None:
    st.sidebar.success("✅ Forecast: Complete")
else:
    st.sidebar.error("❌ Forecast: Pending")

if st.session_state['supply_df'] is not None:
    st.sidebar.success("✅ Supply: Complete")
else:
    st.sidebar.error("❌ Supply: Pending")

if 'optimization_result' in st.session_state:
    st.sidebar.success("✅ Optimization: Complete")
else:
    st.sidebar.error("❌ Optimization: Pending")
