import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- Enhanced Charting Functions ---
def create_cost_vs_profit_chart(cost, profit):
    """Generates an enhanced bar chart to visualize cost vs. profit."""
    labels = ['Cost', 'Profit']
    values = [cost, profit]
    total = sum(values)
    percentages = [f'{(v/total)*100:.1f}%' if total > 0 else '0%' for v in values]
    dollar_values = [f'${v:.2f}' for v in values]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(labels, values, color=['#E74C3C', '#27AE60'], edgecolor='white', linewidth=2)

    # Enhanced styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.tick_params(bottom=False, left=False)
    ax.set_yticklabels(labels, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    ax.xaxis.set_major_formatter('${x:,.0f}')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')
    
    # Grid for better readability
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value and percentage labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            # Dollar value
            ax.text(width / 2, bar.get_y() + bar.get_height()/2,
                    dollar_values[i],
                    ha='center', va='center',
                    fontweight='bold', color='white', fontsize=13)
            # Percentage
            ax.text(width + max(values) * 0.02, bar.get_y() + bar.get_height()/2,
                    percentages[i],
                    ha='left', va='center',
                    fontweight='normal', color='#666', fontsize=11)

    plt.title('Monthly Cost vs Profit Per User', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_revenue_projection_chart(monthly_price, months=12):
    """Creates a revenue projection chart for different user counts."""
    user_counts = [10, 25, 50, 100, 250, 500]
    months_range = np.arange(1, months + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(user_counts)))
    
    for i, users in enumerate(user_counts):
        monthly_revenue = users * monthly_price
        cumulative_revenue = monthly_revenue * months_range
        ax.plot(months_range, cumulative_revenue, 
                marker='o', markersize=6, linewidth=2.5,
                label=f'{users} users', color=colors[i])
    
    ax.set_xlabel('Months', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=12, fontweight='bold')
    ax.set_title('Revenue Projections by User Count', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    # Add background color zones
    ax.axhspan(0, 100000, alpha=0.1, color='red', label='Startup')
    ax.axhspan(100000, 500000, alpha=0.1, color='yellow', label='Growth')
    ax.axhspan(500000, ax.get_ylim()[1], alpha=0.1, color='green', label='Scale')
    
    plt.tight_layout()
    return fig

def create_breakeven_chart(fixed_costs, variable_cost, price_per_user):
    """Creates a break-even analysis chart."""
    max_users = int(fixed_costs / (price_per_user - variable_cost) * 2) if price_per_user > variable_cost else 100
    users = np.linspace(0, max_users, 100)
    
    revenue = users * price_per_user
    total_costs = fixed_costs + (users * variable_cost)
    profit = revenue - total_costs
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(users, revenue, label='Revenue', color='#27AE60', linewidth=3)
    ax.plot(users, total_costs, label='Total Costs', color='#E74C3C', linewidth=3)
    ax.fill_between(users, revenue, total_costs, 
                     where=(revenue >= total_costs), 
                     color='#27AE60', alpha=0.3, label='Profit Zone')
    ax.fill_between(users, revenue, total_costs, 
                     where=(revenue < total_costs), 
                     color='#E74C3C', alpha=0.3, label='Loss Zone')
    
    # Mark break-even point
    if price_per_user > variable_cost:
        breakeven_users = fixed_costs / (price_per_user - variable_cost)
        ax.axvline(x=breakeven_users, color='#3498DB', linestyle='--', linewidth=2)
        ax.text(breakeven_users, ax.get_ylim()[1] * 0.9, 
                f'Break-even: {breakeven_users:.0f} users', 
                ha='center', fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498DB', alpha=0.7, edgecolor='none'),
                color='white')
    
    ax.set_xlabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amount ($)', fontsize=12, fontweight='bold')
    ax.set_title('Break-Even Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    plt.tight_layout()
    return fig

# --- Core Calculation Functions ---
def calculate_tokens(total_minutes, words_per_minute):
    """Estimates total tokens from meeting duration and speaking rate."""
    total_words = total_minutes * words_per_minute
    return total_words / 0.75

def calculate_cost(input_tokens, output_tokens, model_pricing):
    """Calculates the cost based on input and output tokens."""
    input_cost = (input_tokens / 1_000_000) * model_pricing["input_cost_per_million"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output_cost_per_million"]
    return input_cost + output_cost

def calculate_price(cost, profit_margin):
    """Calculates the final price based on cost and desired profit margin."""
    if profit_margin < 100:
        return cost / (1 - (profit_margin / 100))
    return cost

def calculate_market_size(tam, sam_percentage, som_percentage):
    """Calculate SAM and SOM from TAM."""
    sam = tam * (sam_percentage / 100)
    som = sam * (som_percentage / 100)
    return sam, som

# --- Streamlit Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="LLM Profitability Calculator Pro",
    page_icon="📈"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 LLM Profitability Calculator Pro")
st.markdown("""
Advanced profitability modeling for AI-powered meeting analysis services. 
This enhanced calculator includes break-even analysis, market sizing, and competitive pricing strategies.
""")

# --- Initialize Session State ---
if 'scenario_comparison' not in st.session_state:
    st.session_state.scenario_comparison = []

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # --- Model Selection ---
    st.subheader("🤖 LLM Model Selection")
    model_options = {
        "GPT-4o": {
            "input_cost_per_million": 2.50,
            "output_cost_per_million": 10.00
        },
        "GPT-4o-mini": {
            "input_cost_per_million": 0.15,
            "output_cost_per_million": 0.60
        },
        "Claude 3.5 Sonnet": {
            "input_cost_per_million": 3.00,
            "output_cost_per_million": 15.00
        },
        "Claude 3.5 Haiku": {
            "input_cost_per_million": 0.25,
            "output_cost_per_million": 1.25
        },
        "Gemini 1.5 Pro": {
            "input_cost_per_million": 1.25,
            "output_cost_per_million": 5.00
        },
        "Custom Model": {
            "input_cost_per_million": 0.00,
            "output_cost_per_million": 0.00
        }
    }
    
    selected_model_name = st.selectbox("Select LLM Model", list(model_options.keys()))
    
    if selected_model_name == "Custom Model":
        col1, col2 = st.columns(2)
        with col1:
            custom_input_cost = st.number_input("Input Cost ($/M tokens)", 0.0, 100.0, 1.0, 0.1)
        with col2:
            custom_output_cost = st.number_input("Output Cost ($/M tokens)", 0.0, 100.0, 5.0, 0.1)
        selected_model_pricing = {
            "input_cost_per_million": custom_input_cost,
            "output_cost_per_million": custom_output_cost
        }
    else:
        selected_model_pricing = model_options[selected_model_name]
    
    st.markdown("---")
    
    # --- Meeting Parameters ---
    st.subheader("📊 Meeting Parameters")
    meeting_duration_minutes = st.slider("Avg Meeting Duration (min)", 15, 120, 45, 5)
    words_per_minute = st.slider("Speaking Rate (words/min)", 100, 200, 150, 5)
    
    # Meeting frequency selection
    meeting_frequency_type = st.radio(
        "Meeting Frequency Input",
        ["Meetings per Month", "Average Meetings per Week"],
        horizontal=True
    )
    
    if meeting_frequency_type == "Meetings per Month":
        meetings_per_month = st.number_input("Meetings/User/Month", 1, 100, 35, 1)
    else:
        meetings_per_week = st.number_input("Avg Meetings/User/Week", 1, 25, 8, 1)
        meetings_per_month = int(meetings_per_week * 4.33)  # Convert to monthly (52 weeks / 12 months)
        st.caption(f"Equivalent to ~{meetings_per_month} meetings per month")
    
    output_ratio = st.slider("Output Size (% of input)", 5, 100, 25, 5)
    
    st.markdown("---")
    
    # --- Business Model ---
    st.subheader("💼 Business Model")
    profit_margin_goal = st.slider("Target Profit Margin (%)", 0, 90, 40, 5)
    
    # Fixed costs
    st.write("**Monthly Fixed Costs**")
    col1, col2 = st.columns(2)
    with col1:
        infrastructure_cost = st.number_input("Infrastructure ($)", 0, 50000, 5000, 500)
        development_cost = st.number_input("Development ($)", 0, 100000, 20000, 1000)
    with col2:
        marketing_cost = st.number_input("Marketing ($)", 0, 50000, 10000, 500)
        other_fixed_cost = st.number_input("Other Fixed ($)", 0, 50000, 5000, 500)
    
    total_fixed_costs = infrastructure_cost + development_cost + marketing_cost + other_fixed_cost
    
    st.markdown("---")
    
    # --- Market Analysis ---
    st.subheader("🌍 Market Analysis")
    tam = st.number_input("Total Addressable Market ($M)", 100, 10000, 1000, 100) * 1_000_000
    sam_percentage = st.slider("SAM (% of TAM)", 5, 50, 20, 5)
    som_percentage = st.slider("SOM (% of SAM)", 1, 20, 5, 1)
    
    # Competitive pricing
    st.write("**Competitive Landscape**")
    competitor_price = st.number_input("Avg Competitor Price ($/user/month)", 5, 200, 50, 5)

# --- Main Calculations ---
# Token calculations
input_tokens_per_meeting = calculate_tokens(meeting_duration_minutes, words_per_minute)
output_tokens_per_meeting = input_tokens_per_meeting * (output_ratio / 100)

# Cost calculations
cost_per_meeting = calculate_cost(input_tokens_per_meeting, output_tokens_per_meeting, selected_model_pricing)
monthly_cost_per_user = cost_per_meeting * meetings_per_month

# Pricing calculations
price_per_user_per_month = calculate_price(monthly_cost_per_user, profit_margin_goal)
profit_per_user_per_month = price_per_user_per_month - monthly_cost_per_user

# Market calculations
sam, som = calculate_market_size(tam, sam_percentage, som_percentage)
potential_users = som / (price_per_user_per_month * 12) if price_per_user_per_month > 0 else 0

# --- Main Dashboard ---
# Key Metrics Row
st.header("🎯 Key Metrics Dashboard")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Suggested Price/User/Month",
        f"${price_per_user_per_month:.2f}",
        f"{((price_per_user_per_month / competitor_price) - 1) * 100:.1f}% vs competition" if competitor_price > 0 else "N/A"
    )

with col2:
    st.metric(
        "Gross Margin",
        f"{profit_margin_goal}%",
        f"${profit_per_user_per_month:.2f} profit/user"
    )

with col3:
    st.metric(
        "Break-even Users",
        f"{int(total_fixed_costs / profit_per_user_per_month):,}" if profit_per_user_per_month > 0 else "N/A",
        f"${total_fixed_costs:,.0f} fixed costs"
    )

with col4:
    st.metric(
        "Market Potential",
        f"{int(potential_users):,} users",
        f"${som/1_000_000:.1f}M SOM"
    )

st.markdown("---")

# --- Pricing Tiers ---
st.header("💳 Suggested Pricing Tiers")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; height: 200px;'>
        <h3 style='color: white;'>Starter</h3>
        <h1 style='color: white;'>${:.2f}</h1>
        <p>Per user/month</p>
        <ul>
            <li>Up to 20 meetings/month</li>
            <li>Basic summaries</li>
            <li>Email support</li>
        </ul>
    </div>
    """.format(price_per_user_per_month * 0.8), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; color: white; height: 200px;'>
        <h3 style='color: white;'>Professional</h3>
        <h1 style='color: white;'>${:.2f}</h1>
        <p>Per user/month</p>
        <ul>
            <li>Unlimited meetings</li>
            <li>Advanced analytics</li>
            <li>Priority support</li>
        </ul>
    </div>
    """.format(price_per_user_per_month), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 15px; color: white; height: 200px;'>
        <h3 style='color: white;'>Enterprise</h3>
        <h1 style='color: white;'>Custom</h1>
        <p>Contact sales</p>
        <ul>
            <li>Volume discounts</li>
            <li>Custom integrations</li>
            <li>Dedicated support</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Visualizations ---
st.markdown("---")
st.header("📊 Financial Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Cost Breakdown", "Revenue Projections", "Break-even Analysis", "Scenario Planning"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        cost_profit_chart = create_cost_vs_profit_chart(monthly_cost_per_user, profit_per_user_per_month)
        st.pyplot(cost_profit_chart)
    
    with col2:
        st.subheader("Cost Structure Analysis")
        
        # Create a detailed cost breakdown
        cost_data = {
            'Cost Component': ['LLM API Costs', 'Infrastructure', 'Development', 'Marketing', 'Other'],
            'Monthly Cost': [
                monthly_cost_per_user,
                infrastructure_cost / max(potential_users, 1),
                development_cost / max(potential_users, 1),
                marketing_cost / max(potential_users, 1),
                other_fixed_cost / max(potential_users, 1)
            ]
        }
        cost_df = pd.DataFrame(cost_data)
        cost_df['Percentage'] = (cost_df['Monthly Cost'] / cost_df['Monthly Cost'].sum() * 100).round(1)
        
        st.dataframe(cost_df.style.format({'Monthly Cost': '${:.2f}', 'Percentage': '{:.1f}%'}))
        
        # Unit economics
        st.subheader("Unit Economics")
        st.metric("Customer Acquisition Cost (CAC)", f"${marketing_cost / max(potential_users/12, 1):.2f}")
        st.metric("Customer Lifetime Value (CLV)", f"${price_per_user_per_month * 24:.2f}", "Assuming 24-month retention")
        st.metric("LTV:CAC Ratio", f"{(price_per_user_per_month * 24) / max(marketing_cost / max(potential_users/12, 1), 1):.1f}x")

with tab2:
    revenue_chart = create_revenue_projection_chart(price_per_user_per_month)
    st.pyplot(revenue_chart)
    
    # Revenue milestones
    st.subheader("Revenue Milestones")
    milestones = [10000, 50000, 100000, 500000, 1000000]
    milestone_data = []
    
    for milestone in milestones:
        users_needed = milestone / price_per_user_per_month if price_per_user_per_month > 0 else 0
        milestone_data.append({
            'Monthly Revenue': f'${milestone:,}',
            'Users Needed': f'{int(users_needed):,}',
            'As % of SOM': f'{(users_needed * price_per_user_per_month * 12 / som * 100):.1f}%' if som > 0 else 'N/A'
        })
    
    milestone_df = pd.DataFrame(milestone_data)
    st.dataframe(milestone_df)

with tab3:
    breakeven_chart = create_breakeven_chart(total_fixed_costs, monthly_cost_per_user, price_per_user_per_month)
    st.pyplot(breakeven_chart)
    
    # Break-even metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Break-even Analysis")
        if profit_per_user_per_month > 0:
            breakeven_users = total_fixed_costs / profit_per_user_per_month
            breakeven_revenue = breakeven_users * price_per_user_per_month
            months_to_breakeven = breakeven_users / (potential_users / 12) if potential_users > 0 else float('inf')
            
            st.metric("Users to Break-even", f"{int(breakeven_users):,}")
            st.metric("Revenue to Break-even", f"${breakeven_revenue:,.0f}")
            st.metric("Estimated Time to Break-even", f"{months_to_breakeven:.1f} months" if months_to_breakeven < 100 else "N/A")
    
    with col2:
        st.subheader("Sensitivity Analysis")
        st.write("Impact of 10% change in key variables:")
        
        # Price sensitivity
        price_10_up = price_per_user_per_month * 1.1
        price_10_down = price_per_user_per_month * 0.9
        
        sensitivity_data = {
            'Variable': ['Price +10%', 'Price -10%', 'Costs +10%', 'Costs -10%'],
            'New Break-even': [
                int(total_fixed_costs / (price_10_up - monthly_cost_per_user)),
                int(total_fixed_costs / (price_10_down - monthly_cost_per_user)) if price_10_down > monthly_cost_per_user else 'N/A',
                int(total_fixed_costs / (price_per_user_per_month - monthly_cost_per_user * 1.1)) if price_per_user_per_month > monthly_cost_per_user * 1.1 else 'N/A',
                int(total_fixed_costs / (price_per_user_per_month - monthly_cost_per_user * 0.9))
            ]
        }
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        st.dataframe(sensitivity_df)

with tab4:
    st.subheader("Scenario Comparison Tool")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        scenario_name = st.text_input("Scenario Name", "Current")
    with col2:
        if st.button("Save Current Scenario"):
            scenario = {
                'Name': scenario_name,
                'Model': selected_model_name,
                'Price/User': price_per_user_per_month,
                'Profit Margin': profit_margin_goal,
                'Break-even Users': int(total_fixed_costs / profit_per_user_per_month) if profit_per_user_per_month > 0 else 'N/A',
                'Annual Revenue at 100 Users': price_per_user_per_month * 100 * 12
            }
            st.session_state.scenario_comparison.append(scenario)
    with col3:
        if st.button("Clear All Scenarios"):
            st.session_state.scenario_comparison = []
    
    if st.session_state.scenario_comparison:
        scenario_df = pd.DataFrame(st.session_state.scenario_comparison)
        st.dataframe(scenario_df.style.format({
            'Price/User': '${:.2f}',
            'Profit Margin': '{:.0f}%',
            'Annual Revenue at 100 Users': '${:,.0f}'
        }))

# --- Detailed Calculations ---
with st.expander("🔍 Detailed Calculations & Assumptions"):
    tab1, tab2, tab3 = st.tabs(["Token Analysis", "Cost Breakdown", "Market Sizing"])
    
    with tab1:
        st.subheader("Token Calculation Details")
        col1, col2, col3 = st.columns(3)
        col1.metric("Words per Meeting", f"{meeting_duration_minutes * words_per_minute:,}")
        col2.metric("Input Tokens per Meeting", f"{input_tokens_per_meeting:,.0f}")
        col3.metric("Output Tokens per Meeting", f"{output_tokens_per_meeting:,.0f}")
        
        st.write("**Monthly Token Usage per User:**")
        monthly_input_tokens = input_tokens_per_meeting * meetings_per_month
        monthly_output_tokens = output_tokens_per_meeting * meetings_per_month
        
        token_data = {
            'Token Type': ['Input Tokens', 'Output Tokens', 'Total Tokens'],
            'Per Meeting': [f"{input_tokens_per_meeting:,.0f}", f"{output_tokens_per_meeting:,.0f}", f"{input_tokens_per_meeting + output_tokens_per_meeting:,.0f}"],
            'Per Month': [f"{monthly_input_tokens:,.0f}", f"{monthly_output_tokens:,.0f}", f"{monthly_input_tokens + monthly_output_tokens:,.0f}"],
            'Cost': [f"${(monthly_input_tokens / 1_000_000) * selected_model_pricing['input_cost_per_million']:.2f}",
                    f"${(monthly_output_tokens / 1_000_000) * selected_model_pricing['output_cost_per_million']:.2f}",
                    f"${monthly_cost_per_user:.2f}"]
        }
        st.dataframe(pd.DataFrame(token_data))
    
    with tab2:
        st.subheader("Complete Cost Structure")
        st.write(f"**Selected Model:** {selected_model_name}")
        st.write(f"- Input: ${selected_model_pricing['input_cost_per_million']:.2f} per million tokens")
        st.write(f"- Output: ${selected_model_pricing['output_cost_per_million']:.2f} per million tokens")
        
        st.write("**Variable Costs (per user per month):**")
        st.write(f"- LLM API Costs: ${monthly_cost_per_user:.2f}")
        st.write(f"- Estimated Support Cost: ${monthly_cost_per_user * 0.1:.2f}")
        st.write(f"- Total Variable Cost: ${monthly_cost_per_user * 1.1:.2f}")
        
        st.write("**Fixed Costs (monthly):**")
        st.write(f"- Infrastructure: ${infrastructure_cost:,}")
        st.write(f"- Development: ${development_cost:,}")
        st.write(f"- Marketing: ${marketing_cost:,}")
        st.write(f"- Other: ${other_fixed_cost:,}")
        st.write(f"- Total Fixed Costs: ${total_fixed_costs:,}")
    
    with tab3:
        st.subheader("Market Size Calculations")
        market_data = {
            'Market Segment': ['TAM', 'SAM', 'SOM'],
            'Value': [f"${tam/1_000_000:.1f}M", f"${sam/1_000_000:.1f}M", f"${som/1_000_000:.1f}M"],
            'Description': [
                'Total Addressable Market',
                f'{sam_percentage}% of TAM - Serviceable Addressable Market',
                f'{som_percentage}% of SAM - Serviceable Obtainable Market'
            ],
            'Potential Users': [
                f"{int(tam / (price_per_user_per_month * 12)):,}",
                f"{int(sam / (price_per_user_per_month * 12)):,}",
                f"{int(potential_users):,}"
            ]
        }
        st.dataframe(pd.DataFrame(market_data))

# --- Export Functionality ---
st.markdown("---")
st.header("📥 Export Results")

col1, col2 = st.columns(2)
with col1:
    # Create summary report
    summary = f"""
# LLM Profitability Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary
- **Recommended Price:** ${price_per_user_per_month:.2f} per user/month
- **Profit Margin:** {profit_margin_goal}%
- **Break-even:** {int(total_fixed_costs / profit_per_user_per_month) if profit_per_user_per_month > 0 else 'N/A'} users
- **Market Opportunity:** ${som/1_000_000:.1f}M SOM

## Model Configuration
- **LLM Model:** {selected_model_name}
- **Average Meeting Duration:** {meeting_duration_minutes} minutes
- **Meetings per User per Month:** {meetings_per_month}

## Financial Metrics
- **Cost per Meeting:** ${cost_per_meeting:.4f}
- **Monthly Cost per User:** ${monthly_cost_per_user:.2f}
- **Monthly Profit per User:** ${profit_per_user_per_month:.2f}
- **Total Fixed Costs:** ${total_fixed_costs:,}

## Market Analysis
- **TAM:** ${tam/1_000_000:.1f}M
- **SAM:** ${sam/1_000_000:.1f}M ({sam_percentage}% of TAM)
- **SOM:** ${som/1_000_000:.1f}M ({som_percentage}% of SAM)
- **Potential Users in SOM:** {int(potential_users):,}
    """
    
    st.download_button(
        label="📄 Download Summary Report",
        data=summary,
        file_name=f"llm_profitability_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

with col2:
    # Create CSV export
    csv_data = {
        'Metric': [
            'Price per User per Month',
            'Cost per User per Month',
            'Profit per User per Month',
            'Profit Margin',
            'Break-even Users',
            'Total Fixed Costs',
            'TAM',
            'SAM',
            'SOM',
            'Potential Users'
        ],
        'Value': [
            price_per_user_per_month,
            monthly_cost_per_user,
            profit_per_user_per_month,
            profit_margin_goal,
            int(total_fixed_costs / profit_per_user_per_month) if profit_per_user_per_month > 0 else 0,
            total_fixed_costs,
            tam,
            sam,
            som,
            int(potential_users)
        ]
    }
    
    csv_df = pd.DataFrame(csv_data)
    csv_string = csv_df.to_csv(index=False)
    
    st.download_button(
        label="📊 Download Data (CSV)",
        data=csv_string,
        file_name=f"llm_profitability_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.info("""
**Disclaimer:** This calculator provides estimates based on the inputs provided. Actual costs and revenues may vary based on numerous factors including 
but not limited to: actual usage patterns, API pricing changes, operational efficiency, market conditions, and competition. Consider this tool as a 
starting point for financial planning rather than a definitive prediction.
""")

st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>LLM Profitability Calculator Pro v2.0 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)