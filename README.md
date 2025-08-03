# 🥛 Dairy Analytics Platform

A comprehensive dairy operations optimization platform built with Streamlit, featuring demand forecasting, supply chain simulation, multi-plant analysis, and **cost & profit optimization**.

## 🌟 Features Overview

### 🏠 Core Analytics
- **Demand Forecasting**: Facebook Prophet-based demand prediction
- **Supply Simulation**: Seasonal and event-based supply modeling
- **Optimization & Analysis**: Linear programming for production optimization

### 🏭 Multi-Plant Operations  
- **Plant Comparison**: Performance benchmarking across facilities
- **Regional Analysis**: Geographic performance insights
- **Demand Pattern Analysis**: Cross-plant demand variability

### 💰 **NEW: Cost & Profit Optimization**
- **Revenue Optimization**: Real M5 pricing data integration (`sell_prices.csv`)
- **Cost Modeling**: Variable and fixed cost simulation
- **Profit Maximization**: ROI-focused linear programming optimization
- **Investment Analysis**: Data-driven capacity expansion recommendations

### 📊 Executive Dashboard
- **KPI Monitoring**: Real-time performance metrics
- **Trend Analysis**: Historical and predictive insights
- **Strategic Reports**: Business intelligence for decision making

## 📁 Project Structure

```
Project2/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/                  # M5 Competition Dataset
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   ├── sell_prices.csv    # 💰 Pricing data for profit optimization
│   └── sample_submission.csv
└── utils/                 # Core modules
    ├── data_loader.py     # Data processing & pricing integration
    ├── forecast.py        # Prophet demand forecasting
    ├── optimizer.py       # Linear programming & profit optimization
    └── plot.py           # Visualization functions
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Project2

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Data Setup
Ensure M5 competition data files are in the `data/` directory:
- `sales_train_validation.csv` - Historical sales data
- `calendar.csv` - Date mapping and events
- `sell_prices.csv` - Pricing data for profit calculations

## 📊 Navigation Guide

### 1. 🏠 Home - Demand Forecasting
- Historical sales analysis
- Prophet-based demand forecasting
- Statistical summaries and trends

### 2. 🚚 Supply Simulation
- Seasonal supply pattern modeling
- Event-based supply adjustments
- Supply-demand gap analysis

### 3. 🔄 Optimization & Analysis
- Linear programming optimization
- Capacity utilization analysis
- Production schedule optimization

### 4. 🏭 Multi-Plant Analysis
- **Comprehensive Comparison**: Efficiency and performance metrics
- **Regional Analysis**: Geographic performance insights
- **Demand Patterns**: Cross-plant demand analysis

### 5. 💰 Cost & Profit Optimization ⭐ **NEW**
- **Single Plant Analysis**: Individual plant profitability
- **Multi-Plant Comparison**: Profit performance benchmarking
- **ROI Analysis**: Investment decision support

### 6. 📊 Dashboard & Reports
- Executive KPI summary
- Performance trends
- Strategic insights

## 🔧 Technical Architecture

### Data Processing Pipeline
```python
# Load M5 competition data
sales_df = load_sales_data()
calendar_df = load_calendar_data()
prices_df = load_prices_data()  # 💰 New pricing integration

# Process for specific plant
plant_data = preprocess_sales_data_by_plant(sales_df, calendar_df, store_id)
forecast_df = forecast_demand(plant_data)

# Profit optimization
pricing_data = get_item_pricing_data(prices_df, calendar_df, item_id, store_id)
production_costs = simulate_production_costs(plant_capacities)
profit_result = optimize_for_profit(forecast_df, pricing_data, production_costs)
```

### Optimization Models

#### Standard Optimization (Capacity-focused)
```python
# Minimize difference between production and demand
minimize: |production - demand|
subject to: production ≤ capacity
```

#### Profit Optimization (ROI-focused) 💰
```python
# Maximize total profit
maximize: (price × sales) - (variable_cost × production) - fixed_costs
subject to: 
    production ≤ capacity
    sales ≤ min(production, demand)
```

## 💼 Business Applications

### Strategic Planning
- **Investment Decisions**: ROI-based capacity expansion
- **Resource Allocation**: Optimize across multiple plants
- **Performance Benchmarking**: Identify best practices

### Operational Excellence  
- **Production Planning**: Profit-optimized schedules
- **Cost Management**: Variable vs fixed cost analysis
- **Pricing Strategy**: Price-profit sensitivity analysis

### Financial Analysis
- **Profit Margin Optimization**: Maximize margins per SKU
- **Cost Structure Analysis**: Identify efficiency opportunities
- **ROI Tracking**: Monitor investment performance

## 📈 Key Metrics

### Financial KPIs 💰
- **Total Profit**: Revenue - Variable Costs - Fixed Costs
- **Profit Margin %**: (Profit / Revenue) × 100
- **ROI %**: (Profit / Investment) × 100
- **Revenue per Capacity**: Total Revenue / Plant Capacity

### Operational KPIs
- **Capacity Utilization**: (Production / Capacity) × 100
- **Demand Fulfillment**: (Sales / Demand) × 100
- **Supply Utilization**: (Used Supply / Available Supply) × 100

### Performance KPIs
- **Efficiency Score**: Weighted combination of key metrics
- **Cost per Unit**: Variable + Fixed costs per unit produced
- **Price Volatility**: Standard deviation of selling prices

## 🎯 Use Cases

### Dairy Plant Manager
- Monitor daily profit performance
- Optimize production schedules for maximum ROI
- Analyze cost structure and identify savings

### Regional Operations Director
- Compare plant performance across regions
- Identify investment opportunities
- Allocate resources based on profitability

### Executive Leadership
- Strategic investment decisions
- Performance benchmarking
- Financial planning and forecasting

## 📊 Sample Insights

### Profit Analysis Results
```
Plant CA_1 Performance:
- Total Profit: $12,450
- Profit Margin: 18.5%
- ROI: 24.2%
- Capacity Utilization: 87%
- Recommendation: Expand capacity
```

### Multi-Plant Comparison
```
Performance Ranking:
1. CA_1: Efficiency Score 89.2 (High ROI, Good margins)
2. TX_1: Efficiency Score 78.5 (Moderate performance)  
3. WI_1: Efficiency Score 71.3 (High margins, low utilization)
4. FL_1: Efficiency Score 65.8 (Needs cost optimization)
```

## 🔍 Advanced Features

### Cost Modeling
- **Economies of Scale**: Larger plants have lower per-unit costs
- **Fixed vs Variable**: Realistic cost structure simulation
- **Plant-Specific Variations**: Regional cost differences

### Pricing Integration
- **Real M5 Data**: Uses actual `sell_prices.csv` data
- **Weekly to Daily Mapping**: Calendar-based price interpolation
- **Price Volatility Analysis**: Track pricing trends and variations

### Optimization Algorithms
- **Linear Programming**: PuLP-based optimization
- **Multi-Objective**: Balance profit, utilization, and fulfillment
- **Constraint Handling**: Capacity, demand, and cost constraints

## 📚 Data Sources

- **M5 Competition Dataset**: Walmart sales data for forecasting
- **Custom Simulations**: Supply patterns and production costs
- **Industry Benchmarks**: Realistic dairy operation parameters

## 🛠️ Dependencies

- **Streamlit**: Web application framework
- **Prophet**: Time series forecasting
- **PuLP**: Linear programming optimization
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Visualization

## 🚦 Status

✅ **Completed Features:**
- Demand forecasting with Prophet
- Supply simulation and optimization
- Multi-plant analysis and comparison
- **Cost & profit optimization integration** 💰
- Comprehensive visualization dashboards

🔄 **Future Enhancements:**
- Machine learning-based cost prediction
- Real-time data integration
- Advanced forecasting models
- Supply chain risk analysis

## 📞 Support

For questions or issues, please refer to:
- `PROFIT_OPTIMIZATION_README.md` - Detailed profit optimization guide
- Individual module documentation in `utils/` folder
- Streamlit app help sections

---

**Built with ❤️ for dairy industry optimization** 🥛

Transform your dairy operations from capacity-driven to profit-driven with data-driven insights and optimization! 🚀
