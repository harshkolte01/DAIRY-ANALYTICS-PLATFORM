import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from utils.data_downloader import download_data_if_needed
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

st.set_page_config(page_title="Demand Forecasting Dashboard - Academic Project", layout="wide", initial_sidebar_state="collapsed")

# ACADEMIC PROJECT DISCLAIMER - PROMINENT DISPLAY
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.1);
">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 24px; margin-right: 10px;">🎓</span>
        <h2 style="color: white; margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 600;">
            ACADEMIC PROJECT NOTICE
        </h2>
    </div>
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <p style="margin-bottom: 12px; font-size: 16px;">
            <strong style="color: #ffd700;">📚 Educational Purpose:</strong>
            This dashboard demonstrates advanced forecasting methodologies using the industry-standard M5 Competition dataset (2011-2016).
        </p>
        <p style="margin-bottom: 12px; font-size: 16px;">
            <strong style="color: #ffd700;">📊 Data Context:</strong>
            Historical retail data serves as a benchmark for algorithm validation and methodology demonstration.
        </p>
        <p style="margin-bottom: 0; font-size: 16px;">
            <strong style="color: #ffd700;">🎯 Focus:</strong>
            Showcasing technical excellence in machine learning, feature engineering, and optimization techniques applicable to modern business scenarios.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Check and setup data if needed - this will download from Dropbox if files are missing
if not download_data_if_needed():
    st.stop()

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

# Initialize workflow state
if 'workflow_step' not in st.session_state:
    st.session_state['workflow_step'] = 1
if 'completed_steps' not in st.session_state:
    st.session_state['completed_steps'] = set()

# Progress tracking functions
def mark_step_complete(step_number):
    st.session_state['completed_steps'].add(step_number)
    if step_number == st.session_state['workflow_step']:
        st.session_state['workflow_step'] = step_number + 1

def is_step_complete(step_number):
    return step_number in st.session_state['completed_steps']

def can_access_step(step_number):
    return step_number <= st.session_state['workflow_step']

# Main title
st.title("🥛 Demand Forecasting Dashboard - Sequential Workflow")
st.markdown("**Complete each step in order to build a comprehensive forecasting system**")

# Progress bar at the top
st.markdown("### 📊 Workflow Progress")
progress_steps = [
    "🏠 Demand Forecasting",
    "🚚 Supply Simulation", 
    "🔄 Optimization & Analysis",
    "🏭 Multi-Plant Analysis",
    "💰 Cost & Profit Optimization",
    "🤖 Advanced ML Analytics",
    "📊 Dashboard & Reports"
]

progress_cols = st.columns(len(progress_steps))
for i, (col, step_name) in enumerate(zip(progress_cols, progress_steps), 1):
    with col:
        if is_step_complete(i):
            st.success(f"✅ Step {i}")
            st.caption(step_name)
        elif can_access_step(i):
            st.info(f"🔄 Step {i}")
            st.caption(step_name)
        else:
            st.error(f"⏳ Step {i}")
            st.caption(step_name)

st.markdown("---")

# STEP 1: DEMAND FORECASTING
if can_access_step(1):
    st.header("🏠 Step 1: Demand Forecasting")
    st.markdown("**A Comprehensive Forecasting Methodology Using M5 Competition Dataset**")
    
    # Methodology Context
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <h3 style="color: white; margin-top: 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🎯</span>
            Academic Project Overview
        </h3>
        <p style="font-size: 16px; line-height: 1.6; margin-bottom: 15px;">
            This dashboard demonstrates <strong>advanced forecasting methodologies</strong> using the industry-standard M5 Competition dataset.
            While the data is historical (2011-2016), the <strong>techniques are current</strong> and widely used in industry today.
        </p>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-top: 15px;">
            <h4 style="color: #ffd700; margin-top: 0; margin-bottom: 10px;">🎓 Key Learning Objectives:</h4>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>Master Prophet forecasting algorithm (used by Facebook)</li>
                <li>Implement feature engineering and ML pipelines</li>
                <li>Demonstrate production-ready optimization techniques</li>
                <li>Show scalable architecture for real-world deployment</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Age Context
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-left: 4px solid #ffd700;
    ">
        <p style="margin: 0; font-size: 16px; display: flex; align-items: center;">
            <span style="font-size: 20px; margin-right: 10px;">📅</span>
            <strong>Data Context:</strong> This project uses M5 Competition data (2011-2016) for educational purposes.
            The forecasting methodologies demonstrated are current and can be applied to modern datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
                mark_step_complete(1)
                st.success("✅ Forecast completed successfully! You can now proceed to Step 2.")
                st.rerun()
        
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

# STEP 2: SUPPLY SIMULATION
if can_access_step(2):
    st.markdown("---")
    st.header("🚚 Step 2: Supply Simulation")
    
    if not is_step_complete(1):
        st.warning("⚠️ Please complete Step 1 (Demand Forecasting) first.")
    else:
        st.markdown("Configure and simulate milk supply considering seasonal trends, calendar events, and operational constraints.")
        
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
                    mark_step_complete(2)
                    st.success("✅ Supply simulation completed! You can now proceed to Step 3.")
                    st.rerun()
            
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

# STEP 3: OPTIMIZATION & ANALYSIS
if can_access_step(3):
    st.markdown("---")
    st.header("🔄 Step 3: Production Optimization & Gap Analysis")
    
    if not is_step_complete(2):
        st.warning("⚠️ Please complete Step 2 (Supply Simulation) first.")
    else:
        st.markdown("Analyze supply-demand gaps and optimize production schedules under real constraints.")
        
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
                    mark_step_complete(3)
                    st.success("✅ Optimization completed! You can now proceed to Step 4.")
                    st.rerun()
            
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

# STEP 4: MULTI-PLANT ANALYSIS
if can_access_step(4):
    st.markdown("---")
    st.header("🏭 Step 4: Multi-Plant Analysis")
    
    if not is_step_complete(3):
        st.warning("⚠️ Please complete Step 3 (Optimization & Analysis) first.")
    else:
        st.markdown("Compare demand and utilization across multiple store_ids representing different dairy plants or regions.")
        
        # Load available stores and states
        stores_states = get_available_stores_and_states(st.session_state['sales_df'])
        
        if not stores_states:
            st.error("❌ No store/state data found in the dataset.")
        else:
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
            
            if len(selected_stores) >= 2:
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
                        else:
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
                                        
                                        mark_step_complete(4)
                                        st.success("✅ Multi-plant analysis completed! You can now proceed to Step 5.")
                                        
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
                                    mark_step_complete(4)
                                    st.success("✅ Regional analysis completed! You can now proceed to Step 5.")
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
                                    
                                    mark_step_complete(4)
                                    st.success("✅ Demand pattern analysis completed! You can now proceed to Step 5.")
                                    
                                else:
                                    st.error("❌ No forecast data available for demand pattern analysis.")
            else:
                st.warning("⚠️ Please select at least 2 plants for comparison analysis.")

# STEP 5: COST & PROFIT OPTIMIZATION
if can_access_step(5):
    st.markdown("---")
    st.header("💰 Step 5: Cost & Profit Optimization")
    
    if not is_step_complete(4):
        st.warning("⚠️ Please complete Step 4 (Multi-Plant Analysis) first.")
    else:
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
                    else:
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
                            
                            mark_step_complete(5)
                            st.success("✅ Profit analysis completed! You can now proceed to Step 6.")
                            
                        else:
                            st.error("❌ Profit optimization failed. Please check your parameters.")

# STEP 6: ADVANCED ML ANALYTICS
if can_access_step(6):
    st.markdown("---")
    st.header("🤖 Step 6: Advanced ML Analytics")
    
    if not is_step_complete(5):
        st.warning("⚠️ Please complete Step 5 (Cost & Profit Optimization) first.")
    else:
        st.markdown("**Methodology Demonstration: Advanced ML Techniques Using Industry-Standard Dataset**")
        
        # Academic Context for ML Section
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            border-left: 4px solid #ffd700;
        ">
            <p style="margin: 0; font-size: 16px; line-height: 1.6; display: flex; align-items: flex-start;">
                <span style="font-size: 24px; margin-right: 12px; margin-top: -2px;">🎓</span>
                <span>
                    <strong style="color: #ffd700;">Academic Focus:</strong> This section demonstrates advanced machine learning methodologies including
                    ensemble methods, feature engineering, and automated insights generation. The techniques shown are
                    currently used by major tech companies (Meta, Amazon, Netflix) for production forecasting systems.
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
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
                            mark_step_complete(6)
                            st.success("✅ Pre-trained models loaded successfully! You can now proceed to Step 7.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load pre-trained models")
            
            with col2:
                if st.button("🔄 Train New Models", type="secondary"):
                    st.session_state['ml_loader'] = None  # Reset to force new training
                    st.info("Use the training option below to train new models")
        else:
            st.info("💡 No pre-trained models found. You can train models below.")
        
        # Quick Analytics Summary
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
                
                mark_step_complete(6)
                st.success("✅ ML analytics completed! You can now proceed to Step 7.")

# STEP 7: DASHBOARD & REPORTS
if can_access_step(7):
    st.markdown("---")
    st.header("📊 Step 7: Executive Dashboard & Reports")
    
    if not is_step_complete(6):
        st.warning("⚠️ Please complete Step 6 (Advanced ML Analytics) first.")
    else:
        st.markdown("**Academic Project: Comprehensive Analytics Methodology Demonstration**")
        
        # Executive Summary Context
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            <h3 style="color: white; margin-top: 0; display: flex; align-items: center;">
                <span style="margin-right: 10px;">🎯</span>
                Dashboard Purpose
            </h3>
            <p style="font-size: 16px; line-height: 1.6; margin-bottom: 15px;">
                This executive dashboard demonstrates <strong>business intelligence integration</strong> with machine learning predictions.
                The KPIs and metrics shown represent industry-standard approaches to performance monitoring and strategic decision-making.
            </p>
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; margin-top: 15px;">
                <p style="margin: 0; color: #ffd700;">
                    <strong>📚 Educational Value:</strong> Shows how technical ML models translate into actionable business insights.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        mark_step_complete(7)
        st.success("🎉 **Congratulations!** You have completed all steps of the demand forecasting workflow!")
        
        # Final completion message
        st.markdown("---")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
        ">
            <h2 style="color: white; margin-top: 0;">🎉 Workflow Complete!</h2>
            <p style="font-size: 18px; margin-bottom: 0;">
                You have successfully completed all 7 steps of the comprehensive demand forecasting and optimization workflow.
                This demonstrates a complete end-to-end business intelligence and machine learning pipeline.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Sidebar status (simplified)
st.sidebar.markdown("### 🔧 System Status")
completed_count = len(st.session_state['completed_steps'])
st.sidebar.metric("Completed Steps", f"{completed_count}/7")

for i in range(1, 8):
    step_name = progress_steps[i-1]
    if is_step_complete(i):
        st.sidebar.success(f"✅ {step_name}")
    elif can_access_step(i):
        st.sidebar.info(f"🔄 {step_name}")
    else:
        st.sidebar.error(f"⏳ {step_name}")
                
