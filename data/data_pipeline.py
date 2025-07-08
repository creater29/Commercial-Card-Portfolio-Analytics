import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

class CommercialCardDataPipeline:
    
    def __init__(self):
        self.engine = self._create_db_connection()
        self.logger = self._setup_logging()
        
    def _create_db_connection(self):
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        return create_engine(db_url)
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def extract_transaction_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        query = f"""
        SELECT 
            transaction_id,
            card_number_hash,
            customer_id,
            transaction_date,
            transaction_amount,
            merchant_category,
            merchant_name,
            transaction_type,
            authorization_code,
            country_code,
            city,
            created_timestamp
        FROM card_transactions 
        WHERE transaction_date BETWEEN '{start_date}' AND '{end_date}'
        AND transaction_status = 'APPROVED'
        """
        
        self.logger.info(f"Extracting transaction data from {start_date} to {end_date}")
        df = pd.read_sql(query, self.engine)
        self.logger.info(f"Extracted {len(df)} transactions")
        return df
    
    def extract_customer_data(self) -> pd.DataFrame:
        query = """
        SELECT 
            customer_id,
            customer_since_date,
            customer_segment,
            annual_income,
            credit_limit,
            age_group,
            employment_status,
            geographic_region,
            primary_card_type,
            digital_engagement_score
        FROM customer_master
        WHERE status = 'ACTIVE'
        """
        
        self.logger.info("Extracting customer data")
        df = pd.read_sql(query, self.engine)
        self.logger.info(f"Extracted {len(df)} customers")
        return df
    
    def extract_campaign_data(self) -> pd.DataFrame:
        query = """
        SELECT 
            campaign_id,
            customer_id,
            campaign_name,
            campaign_type,
            start_date,
            end_date,
            channel,
            response_flag,
            conversion_flag,
            campaign_cost,
            revenue_generated
        FROM marketing_campaigns
        WHERE start_date >= CURRENT_DATE - INTERVAL '12 months'
        """
        
        self.logger.info("Extracting campaign data")
        df = pd.read_sql(query, self.engine)
        self.logger.info(f"Extracted {len(df)} campaign records")
        return df
    
    def transform_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_month'] = df['transaction_date'].dt.to_period('M')
        df['transaction_year'] = df['transaction_date'].dt.year
        df['transaction_quarter'] = df['transaction_date'].dt.quarter
        df['day_of_week'] = df['transaction_date'].dt.day_name()
        df['is_weekend'] = df['transaction_date'].dt.weekday >= 5
        
        df['amount_bucket'] = pd.cut(df['transaction_amount'], 
                                   bins=[0, 50, 200, 500, 1000, float('inf')],
                                   labels=['Small', 'Medium', 'Large', 'Very Large', 'Premium'])
        
        return df
    
    def create_customer_features(self, transactions_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
        customer_metrics = transactions_df.groupby('customer_id').agg({
            'transaction_amount': ['sum', 'mean', 'count', 'std'],
            'transaction_date': ['min', 'max'],
            'merchant_category': 'nunique'
        }).round(2)
        
        customer_metrics.columns = [
            'total_spend', 'avg_transaction_amount', 'transaction_count', 
            'spend_volatility', 'first_transaction', 'last_transaction', 'unique_categories'
        ]
        
        customer_metrics['days_active'] = (customer_metrics['last_transaction'] - customer_metrics['first_transaction']).dt.days + 1
        customer_metrics['avg_monthly_spend'] = customer_metrics['total_spend'] / (customer_metrics['days_active'] / 30.44)
        customer_metrics['transaction_frequency'] = customer_metrics['transaction_count'] / customer_metrics['days_active'] * 30.44
        
        customer_metrics['recency_days'] = (datetime.now().date() - customer_metrics['last_transaction'].dt.date).dt.days
        
        enhanced_customers = customers_df.merge(customer_metrics, left_on='customer_id', right_index=True, how='left')
        
        enhanced_customers['credit_utilization'] = enhanced_customers['avg_monthly_spend'] / enhanced_customers['credit_limit'] * 100
        enhanced_customers['customer_tenure_months'] = (datetime.now().date() - pd.to_datetime(enhanced_customers['customer_since_date']).dt.date).dt.days / 30.44
        
        return enhanced_customers
    
    def calculate_campaign_metrics(self, campaign_df: pd.DataFrame) -> pd.DataFrame:
        campaign_summary = campaign_df.groupby(['campaign_id', 'campaign_name', 'campaign_type', 'channel']).agg({
            'customer_id': 'count',
            'response_flag': 'sum',
            'conversion_flag': 'sum',
            'campaign_cost': 'first',
            'revenue_generated': 'sum'
        }).reset_index()
        
        campaign_summary.columns = ['campaign_id', 'campaign_name', 'campaign_type', 'channel', 
                                  'total_customers', 'responses', 'conversions', 'campaign_cost', 'revenue']
        
        campaign_summary['response_rate'] = (campaign_summary['responses'] / campaign_summary['total_customers'] * 100).round(2)
        campaign_summary['conversion_rate'] = (campaign_summary['conversions'] / campaign_summary['total_customers'] * 100).round(2)
        campaign_summary['roi'] = ((campaign_summary['revenue'] - campaign_summary['campaign_cost']) / campaign_summary['campaign_cost'] * 100).round(2)
        campaign_summary['cost_per_acquisition'] = (campaign_summary['campaign_cost'] / campaign_summary['conversions']).round(2)
        
        return campaign_summary
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        filepath = f"data/processed/{filename}"
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved processed data to {filepath}")
    
    def run_pipeline(self):
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)
            
            transactions_df = self.extract_transaction_data(start_date.strftime('%Y-%m-%d'), 
                                                          end_date.strftime('%Y-%m-%d'))
            customers_df = self.extract_customer_data()
            campaigns_df = self.extract_campaign_data()
            
            transactions_processed = self.transform_transaction_data(transactions_df)
            customers_enhanced = self.create_customer_features(transactions_processed, customers_df)
            campaigns_summary = self.calculate_campaign_metrics(campaigns_df)
            
            self.save_processed_data(transactions_processed, 'transactions_processed.csv')
            self.save_processed_data(customers_enhanced, 'customers_enhanced.csv')
            self.save_processed_data(campaigns_summary, 'campaigns_summary.csv')
            
            self.logger.info("Data pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = CommercialCardDataPipeline()
    pipeline.run_pipeline()
