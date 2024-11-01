import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths (for testing on local machine)
REFERENCE_PATH = "Reference Pig Data.csv"
#DUMMY_PATH = "Real_Time_Dummy_Pig_Data.csv"
pig_data = pd.read_csv(REFERENCE_PATH)

#def load_data():
  # """Load reference and dummy datasets."""
  #  pig_data = pd.read_csv(REFERENCE_PATH)
  #  dummy_data = pd.read_csv(DUMMY_PATH)
  #  return pig_data, dummy_data

def categorize_age_groups(df):
    """Add an 'Age Group' column based on weeks."""
    bins = [0, 4, 12, 24, 52]  # Example week ranges for age groups
    labels = ['Weaners', 'Growers', 'Finishers', 'Adults']
    df['Age Group'] = pd.cut(df['Week'], bins=bins, labels=labels, right=False)
    return df

#def perform_analysis():
def perform_analysis(pig_data, dummy_data):
    """Perform data analysis and return results as a dictionary."""
    #pig_data, dummy_data = load_data()
    dummy_data = categorize_age_groups(dummy_data)
    predictions = {}

    dummy_data['Feed Cost per kg Gain ($)'] = dummy_data['Average Weekly Feed Cost ($)'] / dummy_data['Average Weekly Weight Gain (kg)']
    # 1. Growth and Feed Efficiency Alerts
    reference_adg_mean = pig_data['ADG (kg/day)'].mean()
    reference_adg_std = pig_data['ADG (kg/day)'].std()
    reference_fcr_mean = pig_data['Average FCR'].mean()
    reference_fcr_std = pig_data['Average FCR'].std()

    predictions['growth_alerts'] = dummy_data[
        (dummy_data['ADG (kg/day)'] < reference_adg_mean - reference_adg_std) | 
        (dummy_data['Average FCR'] > reference_fcr_mean + reference_fcr_std)
    ][['Week', 'ADG (kg/day)', 'Average FCR']].to_dict(orient='records')

    # 2. Cost Efficiency Tracking
    predictions['cost_alerts'] = dummy_data[
        (dummy_data['Average Weekly Feed Cost ($)'] > pig_data['Average  Weekly Feed Cost ($)'].mean()) |
        (dummy_data['Average Cumulative Feed Cost ($)'] > pig_data['Average Cumulative Feed Cost ($)'].mean())
    ][['Week', 'Average Weekly Feed Cost ($)', 'Average Cumulative Feed Cost ($)']].to_dict(orient='records')

    # 3. Environmental Condition Warnings
    predictions['env_alerts'] = dummy_data[
        (dummy_data['Environmental Temp (°C)'] < 26) | 
        (dummy_data['Environmental Temp (°C)'] > 30) |
        (dummy_data['Environmental Humidity (%)'] < 60) |
        (dummy_data['Environmental Humidity (%)'] > 65)
    ][['Week', 'Environmental Temp (°C)', 'Environmental Humidity (%)']].to_dict(orient='records')

    # 4. Health Intervention Insights
    predictions['health_intervention_alerts'] = dummy_data[
        (dummy_data['Health Interventions'] != 'None') &
        (dummy_data['Average Weekly Weight Gain (kg)'] < pig_data['Average Weekly Weight Gain (kg)'].mean())
    ][['Week', 'Health Interventions', 'Average Weekly Weight Gain (kg)']].to_dict(orient='records')

    # Additional Analysis
    dummy_data['Projected Next Week Weight'] = dummy_data['Average End Weight (kg)'] + dummy_data['Average Weekly Weight Gain (kg)'].rolling(window=2).mean().shift(-1)
    dummy_data['Rolling Weekly Feed Cost ($)'] = dummy_data['Average Weekly Feed Cost ($)'].rolling(window=2).mean()
    dummy_data['Rolling Cumulative Feed Cost ($)'] = dummy_data['Average Cumulative Feed Cost ($)'].rolling(window=2).mean()
    dummy_data['Feed Cost per kg Gain ($)'] = dummy_data['Average Weekly Feed Cost ($)'] / dummy_data['Average Weekly Weight Gain (kg)']

    predictions['projected_next_week_weight'] = dummy_data[['Week', 'Projected Next Week Weight']].to_dict(orient='records')
    predictions['rolling_weekly_feed_cost'] = dummy_data[['Week', 'Rolling Weekly Feed Cost ($)']].to_dict(orient='records')
    predictions['feed_cost_per_kg_gain'] = dummy_data[['Week', 'Feed Cost per kg Gain ($)']].to_dict(orient='records')

    # Environmental Impact Analysis
    env_growth_correlation = dummy_data[['Environmental Temp (°C)', 'Environmental Humidity (%)', 'Average Weekly Weight Gain (kg)', 'ADG (kg/day)']].corr()
    predictions['env_growth_correlation'] = env_growth_correlation.reset_index().to_dict(orient='records')

    # Intervention Effectiveness
    intervention_weeks = dummy_data[dummy_data['Health Interventions'] != 'None']['Average Weekly Weight Gain (kg)'].mean()
    non_intervention_weeks = dummy_data[dummy_data['Health Interventions'] == 'None']['Average Weekly Weight Gain (kg)'].mean()
    predictions['intervention_effectiveness'] = {
        "intervention_avg_gain": intervention_weeks,
        "non_intervention_avg_gain": non_intervention_weeks
    }
    
    # Custom Alerts for Anomalies
    custom_alerts = []

    # 1. Sudden Drop in Weight Gain
    # Threshold where the farmer/User should be alerted to make a possible change based on prediction.
    weight_gain_drop_threshold = 0.2  # 20% drop
    dummy_data['Weekly Gain Drop (%)'] = dummy_data['Average Weekly Weight Gain (kg)'].pct_change()
    weight_drop_weeks = dummy_data[dummy_data['Weekly Gain Drop (%)'] < -weight_gain_drop_threshold]
    for _, row in weight_drop_weeks.iterrows():
        custom_alerts.append({
            "Type": "Weight Gain Drop",
            "Week": row['Week'],
            "Drop (%)": f"{row['Weekly Gain Drop (%)'] * 100:.2f}%",
            "Average Weekly Weight Gain (kg)": row['Average Weekly Weight Gain (kg)']
        })

    # 2. Unusual Feed Cost Increase
    # Based on age of pigs the FCR drops requiring the feed cost increase threshold.
    feed_cost_increase_threshold = 0.15  # 15% increase
    dummy_data['Weekly Feed Cost Increase (%)'] = dummy_data['Average Weekly Feed Cost ($)'].pct_change()
    cost_increase_weeks = dummy_data[dummy_data['Weekly Feed Cost Increase (%)'] > feed_cost_increase_threshold]
    for _, row in cost_increase_weeks.iterrows():
        custom_alerts.append({
            "Type": "Feed Cost Increase",
            "Week": row['Week'],
            "Increase (%)": f"{row['Weekly Feed Cost Increase (%)'] * 100:.2f}%",
            "Average Weekly Feed Cost ($)": row['Average Weekly Feed Cost ($)']
        })

    # 3. Temperature or Humidity Spike
    temp_upper_limit = 30  # °C
    temp_lower_limit = 26  # °C
    humidity_upper_limit = 65  # %
    humidity_lower_limit = 60  # %
    env_anomalies = dummy_data[
        (dummy_data['Environmental Temp (°C)'] > temp_upper_limit) |
        (dummy_data['Environmental Temp (°C)'] < temp_lower_limit) |
        (dummy_data['Environmental Humidity (%)'] > humidity_upper_limit) |
        (dummy_data['Environmental Humidity (%)'] < humidity_lower_limit)
    ]
    for _, row in env_anomalies.iterrows():
        custom_alerts.append({
            "Type": "Environmental Anomaly",
            "Week": row['Week'],
            "Temperature (°C)": row['Environmental Temp (°C)'],
            "Humidity (%)": row['Environmental Humidity (%)']
        })

    # 4. High Feed Cost per kg Gain
    high_feed_cost_threshold = 2.0  # $ per kg gain
    high_feed_cost_weeks = dummy_data[dummy_data['Feed Cost per kg Gain ($)'] > high_feed_cost_threshold]
    for _, row in high_feed_cost_weeks.iterrows():
        custom_alerts.append({
            "Type": "High Feed Cost per kg Gain",
            "Week": row['Week'],
            "Feed Cost per kg Gain ($)": row['Feed Cost per kg Gain ($)']
        })

    # 5. Drop in Growth During Health Interventions
    intervention_drop_weeks = dummy_data[
        (dummy_data['Health Interventions'] != 'None') &
        (dummy_data['Weekly Gain Drop (%)'] < -weight_gain_drop_threshold)
    ]
    for _, row in intervention_drop_weeks.iterrows():
        custom_alerts.append({
            "Type": "Growth Drop During Intervention",
            "Week": row['Week'],
            "Health Intervention": row['Health Interventions'],
            "Drop (%)": f"{row['Weekly Gain Drop (%)'] * 100:.2f}%",
            "Average Weekly Weight Gain (kg)": row['Average Weekly Weight Gain (kg)']
        })

    predictions['custom_alerts'] = custom_alerts

    return predictions

    # Additional Visuals and Analyses

    # 1. Weekly Average Feed Cost Trend
    plt.figure(figsize=(10, 5))
    plt.plot(dummy_data['Week'], dummy_data['Average Weekly Feed Cost ($)'], color='blue', label='Weekly Feed Cost')
    plt.xlabel('Week')
    plt.ylabel('Average Weekly Feed Cost ($)')
    plt.title('Weekly Average Feed Cost Trend')
    plt.legend()
    plt.savefig("Weekly_Average_Feed_Cost_Trend.jpeg", format="jpeg")
    plt.close()

    # 2. Feed Cost per kg Gain over Time
    plt.figure(figsize=(10, 5))
    plt.plot(dummy_data['Week'], dummy_data['Feed Cost per kg Gain ($)'], color='green', label='Feed Cost per kg Gain')
    plt.xlabel('Week')
    plt.ylabel('Feed Cost per kg Gain ($)')
    plt.title('Feed Cost per kg Gain over Time')
    plt.legend()
    plt.savefig("Feed_Cost_per_kg_Gain_Over_Time.jpeg", format="jpeg")
    plt.close()

    # 3. Average Daily Gain (ADG) Comparison by Age Group
    age_group_adg = dummy_data.groupby('Age Group')['ADG (kg/day)'].mean()
    plt.figure(figsize=(8, 6))
    age_group_adg.plot(kind='bar', color='orange')
    plt.xlabel('Age Group')
    plt.ylabel('Average Daily Gain (kg/day)')
    plt.title('Average Daily Gain by Age Group')
    plt.savefig("Average_Daily_Gain_by_Age_Group.jpeg", format="jpeg")
    plt.close()

    # 4. Rolling Weekly Feed Cost Analysis
    plt.figure(figsize=(10, 5))
    plt.plot(dummy_data['Week'], dummy_data['Rolling Weekly Feed Cost ($)'], color='purple', label='Rolling Weekly Feed Cost')
    plt.xlabel('Week')
    plt.ylabel('Rolling Weekly Feed Cost ($)')
    plt.title('Rolling Weekly Feed Cost Analysis')
    plt.legend()
    plt.savefig("Rolling_Weekly_Feed_Cost_Analysis.jpeg", format="jpeg")
    plt.close()

    # 5. Intervention Effectiveness on Weekly Weight Gain
    intervention_effectiveness = dummy_data.groupby('Health Interventions')['Average Weekly Weight Gain (kg)'].mean()
    plt.figure(figsize=(8, 6))
    intervention_effectiveness.plot(kind='bar', color='skyblue')
    plt.xlabel('Health Intervention')
    plt.ylabel('Average Weekly Weight Gain (kg)')
    plt.title('Effectiveness of Health Interventions on Weight Gain')
    plt.savefig("Intervention_Effectiveness_on_Weight_Gain.jpeg", format="jpeg")
    plt.close()

    # 6. Cumulative Feed Cost Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(dummy_data['Week'], dummy_data['Average Cumulative Feed Cost ($)'], color='red', label='Cumulative Feed Cost')
    plt.xlabel('Week')
    plt.ylabel('Average Cumulative Feed Cost ($)')
    plt.title('Cumulative Feed Cost Over Time')
    plt.legend()
    plt.savefig("Cumulative_Feed_Cost_Over_Time.jpeg", format="jpeg")
    plt.close()

    # 7. Temperature and Humidity Impact on Weekly Weight Gain
    plt.figure(figsize=(10, 6))
    plt.scatter(dummy_data['Environmental Temp (°C)'], dummy_data['Average Weekly Weight Gain (kg)'],
                s=dummy_data['Environmental Humidity (%)']*2, alpha=0.5, c=dummy_data['Environmental Temp (°C)'], cmap='coolwarm')
    plt.colorbar(label='Temperature (°C)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Average Weekly Weight Gain (kg)')
    plt.title('Temperature and Humidity Impact on Weekly Weight Gain')
    plt.savefig("Temp_Humidity_vs_Weight_Gain.jpeg", format="jpeg")
    plt.close()

    return predictions

def display_results(predictions):
    """Display analysis results in table format in the terminal."""
    print("\n=== Growth and Feed Efficiency Alerts ===")
    print(tabulate(predictions['growth_alerts'], headers="keys", tablefmt="grid"))

    print("\n=== Cost Efficiency Alerts ===")
    print(tabulate(predictions['cost_alerts'], headers="keys", tablefmt="grid"))

    print("\n=== Environmental Condition Alerts ===")
    print(tabulate(predictions['env_alerts'], headers="keys", tablefmt="grid"))

    print("\n=== Health Intervention Effectiveness Alerts ===")
    print(tabulate(predictions['health_intervention_alerts'], headers="keys", tablefmt="grid"))

    print("\n=== Projected Next Week Weight ===")
    print(tabulate(predictions['projected_next_week_weight'], headers="keys", tablefmt="grid"))

    print("\n=== Rolling Weekly Feed Cost ($) ===")
    print(tabulate(predictions['rolling_weekly_feed_cost'], headers="keys", tablefmt="grid"))

    print("\n=== Feed Cost per kg Gain ($) ===")
    print(tabulate(predictions['feed_cost_per_kg_gain'], headers="keys", tablefmt="grid"))

    print("\n=== Environmental and Growth Correlation ===")
    print(tabulate(predictions['env_growth_correlation'], headers="keys", tablefmt="grid"))

    print("\n=== Intervention Effectiveness ===")
    intervention_effectiveness = [
        ["Intervention Weeks Avg Gain (kg)", predictions['intervention_effectiveness']['intervention_avg_gain']],
        ["Non-Intervention Weeks Avg Gain (kg)", predictions['intervention_effectiveness']['non_intervention_avg_gain']]
    ]
    print(tabulate(intervention_effectiveness, headers=["Metric", "Value"], tablefmt="grid"))
    
     

# Run analysis and display results
if __name__ == '__main__':
    results = perform_analysis()
    display_results(results)

