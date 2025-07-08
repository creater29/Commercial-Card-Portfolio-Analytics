import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.churn_model import ChurnPredictionModel

st.set_page_config(page_title="Commercial Card Portfolio Analytics", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    transactions = pd.read_csv('data/processed/transactions_processed.csv')
    customers = pd.read_csv('data/processed/customers_enhanced.csv')
    campaigns = pd.read_csv('data/processed/campaigns_summary.csv')
    return transactions, customers, campaigns

def create_kpi_metrics(customers_df, transactions_df):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(customers_df)
        st.metric("Total Active Customers", f"{total_customers:,}")
    
    with col2:
        total_spend = customers_df['total_spend'].sum()
        st.metric("Total Portfolio Spend", f"${total_spend/1e6:.1f}M")
    
    with col3:
        avg_clv = customers_df['total_spend'].mean()
        st.metric("Average Customer Value", f"${avg_clv:,.0f}")
    
    with col4:
        active_rate = (customers_df['recency_days'] <= 30).mean() * 100
        st.metric("30-Day Active Rate", f"{active_rate:.1f}%")

def portfolio_overview_tab(customers_df, transactions_df):
    st.header("📊 Portfolio Overview")
    
    create_kpi_metrics(customers_df, transactions_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segment Distribution")
        segment_counts = customers_df['customer_segment'].value_counts()
        fig_pie = px.pie(values=segment_counts.values, names=segment_counts.index,
                        title="Customers by Segment")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Spend Distribution by Segment")
        segment_spend = customers_df.groupby('customer_segment')['total_spend'].sum().reset_index()
        fig_bar = px.bar(segment_spend, x='customer_segment', y='total_spend',
                        title="Total Spend by Customer Segment")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader("Geographic Performance")
    geo_performance = customers_df.groupby('geographic_region').agg({
        'customer_id': 'count',
        'total_spend': 'sum',
        'avg_monthly_spend': 'mean'
    }).reset_index()
    geo_performance.columns = ['Region', 'Customer Count', 'Total Spend', 'Avg Monthly Spend']
    st.dataframe(geo_performance, use_container_width=True)

def customer_analytics_tab(customers_df):
    st.header("👥 Customer Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Lifetime Value Distribution")
        fig_hist = px.histogram(customers_df, x='total_spend', nbins=50,
                               title="Distribution of Customer Lifetime Value")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("Credit Utilization Analysis")
        fig_box = px.box(customers_df, x='customer_segment', y='credit_utilization',
                        title="Credit Utilization by Segment")
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.subheader("RFM Analysis")
    
    customers_df['recency_score'] = pd.qcut(customers_df['recency_days'], 5, labels=[5,4,3,2,1])
    customers_df['frequency_score'] = pd.qcut(customers_df['transaction_frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    customers_df['monetary_score'] = pd.qcut(customers_df['total_spend'], 5, labels=[1,2,3,4,5])
    
    customers_df['rfm_score'] = (customers_df['recency_score'].astype(str) + 
                                customers_df['frequency_score'].astype(str) + 
                                customers_df['monetary_score'].astype(str))
    
    def categorize_rfm(rfm_score):
        if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif rfm_score in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif rfm_score in ['512', '511', '422', '421', '412', '411', '311']:
            return 'Potential Loyalists'
        elif rfm_score in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
            return 'New Customers'
        elif rfm_score in ['155', '154', '144', '214', '215', '115', '114']:
            return 'At Risk'
        elif rfm_score in ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '125', '124']:
            return 'Cannot Lose Them'
        elif rfm_score in ['155', '154', '144', '214', '215', '115', '114', '113']:
            return 'Hibernating'
        else:
            return 'Lost'
    
    customers_df['customer_category'] = customers_df['rfm_score'].apply(categorize_rfm)
    
    rfm_summary = customers_df.groupby('customer_category').agg({
        'customer_id': 'count',
        'total_spend': 'sum',
        'avg_monthly_spend': 'mean'
    }).reset_index()
    
    st.dataframe(rfm_summary, use_container_width=True)

def churn_analysis_tab(customers_df):
    st.header("⚠️ Churn Analysis")
    
    try:
        churn_model = ChurnPredictionModel()
        churn_model.load_model('models/churn_model.joblib')
        
        churn_probabilities = churn_model.predict_churn_probability(customers_df)
        customers_df['churn_probability'] = churn_probabilities
        customers_df['churn_risk'] = pd.cut(customers_df['churn_probability'], 
                                          bins=[0, 0.3, 0.7, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn Risk Distribution")
            churn_dist = customers_df['churn_risk'].value_counts()
            fig_churn = px.pie(values=churn_dist.values, names=churn_dist.index,
                              title="Customers by Churn Risk")
            st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            st.subheader("High Risk Customers by Segment")
            high_risk = customers_df[customers_df['churn_risk'] == 'High']
            if len(high_risk) > 0:
                risk_by_segment = high_risk['customer_segment'].value_counts()
                fig_risk = px.bar(x=risk_by_segment.index, y=risk_by_segment.values,
                                 title="High Risk Customers by Segment")
                st.plotly_chart(fig_risk, use_container_width=True)
        
        st.subheader("Feature Importance for Churn Prediction")
        feature_importance = churn_model.get_feature_importance()
        fig_importance = px.bar(feature_importance.head(10), 
                               x='importance', y='feature', orientation='h',
                               title="Top 10 Churn Prediction Features")
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.subheader("High Risk Customers for Intervention")
        high_risk_customers = customers_df[customers_df['churn_risk'] == 'High'][
            ['customer_id', 'customer_segment', 'total_spend', 'recency_days', 'churn_probability']
        ].sort_values('total_spend', ascending=False)
        st.dataframe(high_risk_customers.head(20), use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Churn model not found. Please train the model first.")
        st.info("Run: `python src/models/churn_model.py` to train the churn prediction model.")

def campaign_analysis_tab(campaigns_df):
    st.header("📢 Campaign Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_response_rate = campaigns_df['response_rate'].mean()
        st.metric("Average Response Rate", f"{avg_response_rate:.1f}%")
    
    with col2:
        avg_conversion_rate = campaigns_df['conversion_rate'].mean()
        st.metric("Average Conversion Rate", f"{avg_conversion_rate:.1f}%")
    
    with col3:
        avg_roi = campaigns_df['roi'].mean()
        st.metric("Average ROI", f"{avg_roi:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campaign Performance by Type")
        campaign_performance = campaigns_df.groupby('campaign_type').agg({
            'response_rate': 'mean',
            'conversion_rate': 'mean',
            'roi': 'mean'
        }).reset_index()
        
        fig_performance = px.bar(campaign_performance, x='campaign_type', y='roi',
                                title="Average ROI by Campaign Type")
        st.plotly_chart(fig_performance, use_container_width=True)
    
    with col2:
        st.subheader("Channel Effectiveness")
        channel_performance = campaigns_df.groupby('channel').agg({
            'response_rate': 'mean',
            'conversion_rate': 'mean',
            'roi': 'mean'
        }).reset_index()
        
        fig_channel = px.scatter(channel_performance, x='response_rate', y='conversion_rate',
                                size='roi', hover_name='channel',
                                title="Channel Performance: Response vs Conversion")
        st.plotly_chart(fig_channel, use_container_width=True)
    
    st.subheader("Top Performing Campaigns")
    top_campaigns = campaigns_df.nlargest(10, 'roi')[
        ['campaign_name', 'campaign_type', 'channel', 'response_rate', 'conversion_rate', 'roi']
    ]
    st.dataframe(top_campaigns, use_container_width=True)

def main():
    st.title("🏦 Commercial Card Portfolio Analytics Dashboard")
    st.markdown("*Advanced Analytics for Commercial Card Performance & Customer Insights*")
    
    with st.sidebar:
        st.header("Navigation")
        st.markdown("---")
        st.markdown("**Dashboard Sections:**")
    
    try:
        transactions_df, customers_df, campaigns_df = load_data()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "👥 Customer Analytics", "⚠️ Churn Analysis", "📢 Campaign Analysis"])
        
        with tab1:
            portfolio_overview_tab(customers_df, transactions_df)
        
        with tab2:
            customer_analytics_tab(customers_df)
        
        with tab3:
            churn_analysis_tab(customers_df)
        
        with tab4:
            campaign_analysis_tab(campaigns_df)
            
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Please run the data pipeline first: `python src/data/data_pipeline.py`")
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Key Metrics:**")
        st.markdown("- Portfolio Performance")
        st.markdown("- Customer Segmentation") 
        st.markdown("- Churn Prediction")
        st.markdown("- Campaign ROI")
        st.markdown("---")
        st.markdown("**Technologies:**")
        st.markdown("- Python, SQL")
        st.markdown("- Machine Learning")
        st.markdown("- Interactive Dashboards")

if __name__ == "__main__":
    main()
