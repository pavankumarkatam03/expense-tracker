import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedExpensePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.best_model = None
        self.model_performance = {}
        self.is_trained = False
        
    def prepare_features(self, df):
        """Enhanced feature engineering"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Time-based features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['quarter'] = df['Date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical features for seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Encode categorical variables
        categorical_columns = ['City', 'Category', 'Payment_Method']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        # Create interaction features
        df['income_expense_ratio'] = df['Expense'] / df['Income']
        df['day_month_interaction'] = df['day_of_week'] * df['day_of_month']
        
        # Lag features (for time series)
        df = self._create_lag_features(df)
        
        # Rolling statistics
        df = self._create_rolling_features(df)
        
        return df
    
    def _create_lag_features(self, df):
        """Create lag features for time series analysis"""
        df_sorted = df.sort_values('Date')
        
        # Group by category and create lag features
        for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month lags
            df_sorted[f'expense_lag_{lag}'] = df_sorted.groupby('Category')['Expense'].shift(lag)
        
        # Fill NaN values with mean
        lag_columns = [col for col in df_sorted.columns if 'lag_' in col]
        for col in lag_columns:
            df_sorted[col].fillna(df_sorted[col].mean(), inplace=True)
        
        return df_sorted
    
    def _create_rolling_features(self, df):
        """Create rolling window statistics"""
        df_sorted = df.sort_values('Date')
        
        # Rolling mean and std for different windows
        windows = [7, 30]  # 1 week, 1 month
        for window in windows:
            df_sorted[f'rolling_mean_{window}'] = df_sorted.groupby('Category')['Expense'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df_sorted[f'rolling_std_{window}'] = df_sorted.groupby('Category')['Expense'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Fill NaN values
        rolling_columns = [col for col in df_sorted.columns if 'rolling_' in col]
        for col in rolling_columns:
            df_sorted[col].fillna(df_sorted[col].mean(), inplace=True)
        
        return df_sorted
    
    def initialize_models(self):
        """Initialize multiple ML models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgb_regressor': xgb.XGBRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'lasso': Lasso(alpha=0.1, random_state=42),
            'ridge': Ridge(alpha=0.1, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
    
    def prepare_lstm_data(self, df, sequence_length=30):
        """Prepare data for LSTM model"""
        # Group by category and date
        df_sorted = df.sort_values('Date')
        categories = df_sorted['Category'].unique()
        
        sequences = []
        targets = []
        
        for category in categories:
            cat_data = df_sorted[df_sorted['Category'] == category].copy()
            cat_data = cat_data.set_index('Date')
            
            # Resample to daily frequency and fill missing values
            cat_data_daily = cat_data.resample('D').mean()
            cat_data_daily['Expense'].fillna(method='ffill', inplace=True)
            cat_data_daily['Income'].fillna(method='ffill', inplace=True)
            
            # Select features for LSTM
            feature_columns = ['Expense', 'Income', 'month_sin', 'month_cos']
            features = cat_data_daily[feature_columns].values
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                sequences.append(features[i:(i + sequence_length)])
                targets.append(features[i + sequence_length][0])  # Predict expense
        
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_models(self, df):
        """Train all models and select the best one"""
        try:
            print("Preparing features...")
            df_processed = self.prepare_features(df)
            
            # Feature columns for traditional models
            feature_columns = [
                'Income', 'day_of_week', 'day_of_month', 'month', 'year', 'quarter',
                'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                'income_expense_ratio', 'day_month_interaction',
                'City_encoded', 'Category_encoded', 'Payment_Method_encoded',
                'expense_lag_1', 'expense_lag_7', 'expense_lag_30',
                'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30'
            ]
            
            X = df_processed[feature_columns]
            y = df_processed['Future_Expense']
            
            # Scale features
            self.scalers['feature'] = StandardScaler()
            X_scaled = self.scalers['feature'].fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Initialize models
            self.initialize_models()
            
            # Train and evaluate each model
            best_score = -np.inf
            self.best_model = None
            
            for name, model in self.models.items():
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                self.model_performance[name] = {
                    'r2_score': r2,
                    'mae': mae,
                    'rmse': rmse
                }
                
                print(f"{name} - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                
                # Update best model
                if r2 > best_score:
                    best_score = r2
                    self.best_model = model
                    self.best_model_name = name
            
            # Train LSTM model
            print("Training LSTM model...")
            try:
                sequences, targets = self.prepare_lstm_data(df_processed)
                if len(sequences) > 0:
                    # Split LSTM data
                    X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
                        sequences, targets, test_size=0.2, random_state=42
                    )
                    
                    # Scale LSTM data
                    self.scalers['lstm'] = StandardScaler()
                    X_lstm_train_scaled = self.scalers['lstm'].fit_transform(
                        X_lstm_train.reshape(-1, X_lstm_train.shape[-1])
                    ).reshape(X_lstm_train.shape)
                    X_lstm_test_scaled = self.scalers['lstm'].transform(
                        X_lstm_test.reshape(-1, X_lstm_test.shape[-1])
                    ).reshape(X_lstm_test.shape)
                    
                    # Build and train LSTM
                    self.models['lstm'] = self.build_lstm_model((X_lstm_train_scaled.shape[1], X_lstm_train_scaled.shape[2]))
                    
                    history = self.models['lstm'].fit(
                        X_lstm_train_scaled, y_lstm_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_lstm_test_scaled, y_lstm_test),
                        verbose=0
                    )
                    
                    # Evaluate LSTM
                    lstm_pred = self.models['lstm'].predict(X_lstm_test_scaled).flatten()
                    lstm_r2 = r2_score(y_lstm_test, lstm_pred)
                    lstm_mae = mean_absolute_error(y_lstm_test, lstm_pred)
                    lstm_rmse = np.sqrt(mean_squared_error(y_lstm_test, lstm_pred))
                    
                    self.model_performance['lstm'] = {
                        'r2_score': lstm_r2,
                        'mae': lstm_mae,
                        'rmse': lstm_rmse
                    }
                    
                    print(f"LSTM - R²: {lstm_r2:.4f}, MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}")
                    
                    if lstm_r2 > best_score:
                        best_score = lstm_r2
                        self.best_model = self.models['lstm']
                        self.best_model_name = 'lstm'
            except Exception as e:
                print(f"LSTM training failed: {str(e)}")
            
            self.is_trained = True
            print(f"\nBest model: {self.best_model_name} with R²: {best_score:.4f}")
            
            # Retrain best model on full dataset
            if self.best_model_name != 'lstm':
                self.best_model.fit(X_scaled, y)
            
            return True
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            self.is_trained = False
            return False
    
    def predict_future_expenses(self, df, months_ahead=6, prediction_type='category'):
        """
        Predict future expenses
        
        Args:
            df: Historical data
            months_ahead: Number of months to predict
            prediction_type: 'category' for category-wise or 'total' for aggregate
        """
        if not self.is_trained:
            print("Models not trained. Using fallback prediction.")
            return self._fallback_prediction(df, months_ahead)
        
        try:
            df_processed = self.prepare_features(df)
            last_date = df_processed['Date'].max()
            
            predictions = []
            
            for i in range(1, months_ahead + 1):
                future_date = last_date + timedelta(days=30 * i)
                
                if prediction_type == 'category':
                    monthly_predictions = self._predict_by_category(df_processed, future_date)
                else:
                    monthly_predictions = self._predict_aggregate(df_processed, future_date)
                
                predictions.append({
                    'month': future_date.strftime('%Y-%m'),
                    'predictions': monthly_predictions,
                    'total_predicted': sum(monthly_predictions.values()) if isinstance(monthly_predictions, dict) else monthly_predictions
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return self._fallback_prediction(df, months_ahead)
    
    def _predict_by_category(self, df_processed, future_date):
        """Predict expenses by category for a specific date"""
        predictions = {}
        categories = df_processed['Category'].unique()
        
        # Prepare base features for the future date
        base_features = {
            'day_of_week': future_date.weekday(),
            'day_of_month': future_date.day,
            'month': future_date.month,
            'year': future_date.year,
            'quarter': (future_date.month - 1) // 3 + 1,
            'is_weekend': int(future_date.weekday() >= 5),
            'month_sin': np.sin(2 * np.pi * future_date.month / 12),
            'month_cos': np.cos(2 * np.pi * future_date.month / 12),
            'day_sin': np.sin(2 * np.pi * future_date.day / 31),
            'day_cos': np.cos(2 * np.pi * future_date.day / 31),
        }
        
        for category in categories:
            try:
                # Get average income and recent expenses for this category
                cat_data = df_processed[df_processed['Category'] == category]
                avg_income = cat_data['Income'].mean()
                recent_expense = cat_data['Expense'].iloc[-1] if len(cat_data) > 0 else cat_data['Expense'].mean()
                
                # Prepare feature vector
                feature_vector = base_features.copy()
                feature_vector['Income'] = avg_income
                feature_vector['income_expense_ratio'] = recent_expense / avg_income if avg_income > 0 else 0
                feature_vector['day_month_interaction'] = feature_vector['day_of_week'] * feature_vector['day_of_month']
                
                # Add encoded categorical values
                for cat_col in ['City', 'Category', 'Payment_Method']:
                    if cat_col in self.label_encoders:
                        encoder = self.label_encoders[cat_col]
                        if category in encoder.classes_:
                            feature_vector[f'{cat_col}_encoded'] = list(encoder.classes_).index(category)
                        else:
                            feature_vector[f'{cat_col}_encoded'] = 0
                
                # Add lag and rolling features (using recent values)
                feature_vector['expense_lag_1'] = recent_expense
                feature_vector['expense_lag_7'] = recent_expense
                feature_vector['expense_lag_30'] = recent_expense
                feature_vector['rolling_mean_7'] = recent_expense
                feature_vector['rolling_std_7'] = cat_data['Expense'].std() if len(cat_data) > 1 else recent_expense * 0.1
                feature_vector['rolling_mean_30'] = recent_expense
                feature_vector['rolling_std_30'] = cat_data['Expense'].std() if len(cat_data) > 1 else recent_expense * 0.1
                
                # Convert to DataFrame and scale
                feature_df = pd.DataFrame([feature_vector])
                feature_columns = [
                    'Income', 'day_of_week', 'day_of_month', 'month', 'year', 'quarter',
                    'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                    'income_expense_ratio', 'day_month_interaction',
                    'City_encoded', 'Category_encoded', 'Payment_Method_encoded',
                    'expense_lag_1', 'expense_lag_7', 'expense_lag_30',
                    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30'
                ]
                
                feature_df = feature_df[feature_columns]
                feature_scaled = self.scalers['feature'].transform(feature_df)
                
                # Make prediction
                if self.best_model_name == 'lstm':
                    # For LSTM, use a different approach
                    prediction = recent_expense * np.random.uniform(0.9, 1.1)
                else:
                    prediction = self.best_model.predict(feature_scaled)[0]
                
                predictions[category] = max(0, prediction)
                
            except Exception as e:
                print(f"Error predicting for category {category}: {str(e)}")
                predictions[category] = cat_data['Expense'].mean() if len(cat_data) > 0 else 0
        
        return predictions
    
    def _predict_aggregate(self, df_processed, future_date):
        """Predict aggregate expenses for a specific date"""
        # Use category-wise prediction and sum
        category_predictions = self._predict_by_category(df_processed, future_date)
        return sum(category_predictions.values())
    
    def _fallback_prediction(self, df, months_ahead):
        """Fallback prediction using simple moving averages"""
        predictions = []
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['year_month'] = df['Date'].dt.to_period('M')
        
        # Calculate category-wise monthly averages with seasonality
        monthly_avg = df.groupby(['year_month', 'Category'])['Expense'].sum().unstack(fill_value=0)
        
        last_date = df['Date'].max()
        
        for i in range(1, months_ahead + 1):
            future_date = last_date + timedelta(days=30 * i)
            future_month = future_date.month
            
            monthly_predictions = {}
            
            for category in monthly_avg.columns:
                # Get historical data for this category
                historical = monthly_avg[category]
                
                if len(historical) > 0:
                    # Apply seasonal adjustment based on month
                    base_prediction = historical.mean()
                    seasonal_factor = self._get_seasonal_factor(historical, future_month)
                    prediction = base_prediction * seasonal_factor
                    
                    # Add some random variation
                    variation = np.random.uniform(0.9, 1.1)
                    monthly_predictions[category] = max(0, prediction * variation)
                else:
                    monthly_predictions[category] = 0
            
            predictions.append({
                'month': future_date.strftime('%Y-%m'),
                'predictions': monthly_predictions,
                'total_predicted': sum(monthly_predictions.values())
            })
        
        return predictions
    
    def _get_seasonal_factor(self, historical_series, target_month):
        """Calculate seasonal adjustment factor"""
        try:
            # Group by month and calculate average
            monthly_avgs = historical_series.groupby(historical_series.index.month).mean()
            overall_avg = historical_series.mean()
            
            if len(monthly_avgs) > 0 and overall_avg > 0:
                if target_month in monthly_avgs.index:
                    return monthly_avgs[target_month] / overall_avg
                else:
                    return 1.0
            else:
                return 1.0
        except:
            return 1.0
    
    def get_model_performance(self):
        """Get performance metrics for all trained models"""
        return self.model_performance
    
    def save_models(self, filepath):
        """Save trained models and preprocessors"""
        if self.is_trained:
            save_data = {
                'models': self.models,
                'scalers': self.scalers,
                'label_encoders': self.label_encoders,
                'best_model_name': self.best_model_name,
                'model_performance': self.model_performance
            }
            joblib.dump(save_data, filepath)
            print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load trained models and preprocessors"""
        try:
            save_data = joblib.load(filepath)
            self.models = save_data['models']
            self.scalers = save_data['scalers']
            self.label_encoders = save_data['label_encoders']
            self.best_model_name = save_data['best_model_name']
            self.model_performance = save_data['model_performance']
            self.best_model = self.models[self.best_model_name]
            self.is_trained = True
            print(f"Models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

# Utility function for quick predictions
def create_advanced_predictor(df):
    """Create and train an advanced predictor"""
    predictor = AdvancedExpensePredictor()
    predictor.train_models(df)
    return predictor

if __name__ == "__main__":
    # Example usage
    from utils.data_processor import DataProcessor
    
    processor = DataProcessor()
    sample_data = processor.generate_sample_data(2000)
    
    predictor = create_advanced_predictor(sample_data)
    
    # Get predictions
    predictions = predictor.predict_future_expenses(sample_data, months_ahead=6)
    
    print("\nFuture Expense Predictions:")
    for pred in predictions:
        print(f"{pred['month']}: ₹{pred['total_predicted']:.2f}")
    
    # Print model performance
    print("\nModel Performance:")
    for model_name, metrics in predictor.get_model_performance().items():
        print(f"{model_name}: R²={metrics['r2_score']:.4f}, MAE={metrics['mae']:.2f}")