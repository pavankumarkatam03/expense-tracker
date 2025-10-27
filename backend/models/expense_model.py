import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta

class ExpensePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare features for the model"""
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create time-based features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        
        # Encode categorical variables
        categorical_columns = ['City', 'Category', 'Payment_Method']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def train_model(self, df):
        """Train the prediction model"""
        try:
            df_processed = self.prepare_features(df)
            
            # Features for training
            feature_columns = ['Income', 'day_of_week', 'day_of_month', 'month', 'year', 
                             'City_encoded', 'Category_encoded', 'Payment_Method_encoded']
            
            X = df_processed[feature_columns]
            y = df_processed['Future_Expense']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Calculate training score
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            print(f"Model trained successfully. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            # Fallback to simple prediction
            self.is_trained = False
    
    def predict_future_expenses(self, df, months_ahead=6):
        """Predict future expenses for the next few months"""
        if not self.is_trained:
            return self._simple_prediction(df, months_ahead)
        
        try:
            df_processed = self.prepare_features(df)
            
            # Generate future dates
            last_date = df_processed['Date'].max()
            future_dates = [last_date + timedelta(days=30*i) for i in range(1, months_ahead+1)]
            
            predictions = []
            for future_date in future_dates:
                # Create feature set for prediction
                future_data = {
                    'Income': df_processed['Income'].mean(),
                    'day_of_week': future_date.weekday(),
                    'day_of_month': future_date.day,
                    'month': future_date.month,
                    'year': future_date.year,
                    'City_encoded': 0,  # Default city
                    'Category_encoded': 0,  # Default category
                    'Payment_Method_encoded': 0  # Default payment method
                }
                
                # Make prediction for each category
                monthly_predictions = {}
                for category in df_processed['Category'].unique():
                    if category in self.label_encoders['Category'].classes_:
                        future_data['Category_encoded'] = list(self.label_encoders['Category'].classes_).index(category)
                        
                        # Convert to DataFrame for prediction
                        future_df = pd.DataFrame([future_data])
                        prediction = self.model.predict(future_df)[0]
                        monthly_predictions[category] = max(0, prediction)
                
                predictions.append({
                    'month': future_date.strftime('%Y-%m'),
                    'predictions': monthly_predictions,
                    'total_predicted': sum(monthly_predictions.values())
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return self._simple_prediction(df, months_ahead)
    
    def _simple_prediction(self, df, months_ahead):
        """Simple prediction based on historical averages"""
        predictions = []
        
        # Calculate monthly averages by category
        df['Date'] = pd.to_datetime(df['Date'])
        df['year_month'] = df['Date'].dt.to_period('M')
        
        monthly_avg = df.groupby(['year_month', 'Category'])['Expense'].sum().groupby('Category').mean()
        
        last_date = df['Date'].max()
        
        for i in range(1, months_ahead + 1):
            future_date = last_date + timedelta(days=30*i)
            monthly_predictions = {}
            
            for category, avg_expense in monthly_avg.items():
                # Add some random variation
                variation = np.random.uniform(0.9, 1.1)
                monthly_predictions[category] = avg_expense * variation
            
            predictions.append({
                'month': future_date.strftime('%Y-%m'),
                'predictions': monthly_predictions,
                'total_predicted': sum(monthly_predictions.values())
            })
        
        return predictions