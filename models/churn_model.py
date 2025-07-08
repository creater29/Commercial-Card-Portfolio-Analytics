import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionModel:
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_performance = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.copy()
        
        feature_df['is_churned'] = (feature_df['recency_days'] > 60).astype(int)
        
        feature_df['spend_trend'] = feature_df.groupby('customer_id')['avg_monthly_spend'].pct_change(periods=3).fillna(0)
        feature_df['utilization_category'] = pd.cut(feature_df['credit_utilization'], 
                                                   bins=[0, 25, 50, 75, 100, float('inf')],
                                                   labels=['Low', 'Medium', 'High', 'Very High', 'Overlimit'])
        
        feature_df['clv_segment'] = pd.qcut(feature_df['total_spend'], q=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
        
        categorical_features = ['customer_segment', 'age_group', 'employment_status', 
                              'geographic_region', 'primary_card_type', 'utilization_category', 'clv_segment']
        
        for col in categorical_features:
            if col in feature_df.columns:
                le = LabelEncoder()
                feature_df[f'{col}_encoded'] = le.fit_transform(feature_df[col].astype(str))
        
        return feature_df
    
    def select_features(self, df: pd.DataFrame) -> list:
        feature_candidates = [
            'customer_tenure_months', 'total_spend', 'avg_transaction_amount', 
            'transaction_count', 'spend_volatility', 'avg_monthly_spend', 
            'transaction_frequency', 'recency_days', 'credit_utilization',
            'unique_categories', 'digital_engagement_score', 'annual_income',
            'customer_segment_encoded', 'age_group_encoded', 'employment_status_encoded',
            'geographic_region_encoded', 'primary_card_type_encoded', 
            'utilization_category_encoded', 'clv_segment_encoded', 'spend_trend'
        ]
        
        available_features = [col for col in feature_candidates if col in df.columns]
        return available_features
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        prepared_df = self.prepare_features(df)
        self.feature_columns = self.select_features(prepared_df)
        
        X = prepared_df[self.feature_columns].fillna(0)
        y = prepared_df['is_churned']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                          random_state=random_state, stratify=y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'XGBoost': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
            'RandomForest': RandomForestClassifier(random_state=random_state),
            'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000)
        }
        
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train if name != 'LogisticRegression' else X_train_scaled, 
                                      y_train, cv=5, scoring='roc_auc')
            
            self.model_performance[name] = {
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if auc_score > best_score:
                best_score = auc_score
                best_model_name = name
                self.model = model
        
        print(f"Best model: {best_model_name} with AUC: {best_score:.4f}")
        return self.model_performance
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict_churn_probability(self, df: pd.DataFrame) -> np.array:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        prepared_df = self.prepare_features(df)
        X = prepared_df[self.feature_columns].fillna(0)
        
        if hasattr(self.model, 'predict_proba'):
            if isinstance(self.model, LogisticRegression):
                X = self.scaler.transform(X)
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def save_model(self, filepath: str):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'performance': self.model_performance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_performance = model_data['performance']
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    df = pd.read_csv('data/processed/customers_enhanced.csv')
    
    churn_model = ChurnPredictionModel()
    performance = churn_model.train_model(df)
    
    feature_importance = churn_model.get_feature_importance()
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    churn_model.save_model('models/churn_model.joblib')
