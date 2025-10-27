import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataProcessor:
    def __init__(self):
        self.cities = ['Hyderabad', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune']
        self.categories = ['Food', 'Rent', 'Travel', 'Health', 'Education', 'Entertainment', 'Others']
        self.payment_methods = ['Cash', 'Credit Card', 'Debit Card', 'UPI', 'Net Banking']
    
    def generate_sample_data(self, num_records=1000):
        """Generate sample expense data"""
        data = []
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 8, 31)
        
        for i in range(num_records):
            # Random date between Jan 2023 and Aug 2024
            random_days = random.randint(0, (end_date - start_date).days)
            date = start_date + timedelta(days=random_days)
            
            # Monthly income (normally distributed around ₹60,000 ± ₹15,000)
            income = max(30000, min(90000, np.random.normal(60000, 15000)))
            
            # Random category
            category = random.choice(self.categories)
            
            # Expense amount based on category
            base_expenses = {
                'Food': (2000, 8000),
                'Rent': (10000, 25000),
                'Travel': (1000, 5000),
                'Health': (500, 3000),
                'Education': (1000, 6000),
                'Entertainment': (500, 4000),
                'Others': (500, 3000)
            }
            
            min_exp, max_exp = base_expenses[category]
            expense = random.uniform(min_exp, max_exp)
            
            # Future expense with ±10-20% variation
            variation = random.uniform(0.8, 1.2)
            future_expense = expense * variation
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Month': date.strftime('%B'),
                'City': random.choice(self.cities),
                'Income': round(income, 2),
                'Category': category,
                'Expense': round(expense, 2),
                'Payment_Method': random.choice(self.payment_methods),
                'Future_Expense': round(future_expense, 2)
            })
        
        return pd.DataFrame(data)
    
    def load_sample_data(self):
        """Load or generate sample data"""
        try:
            df = pd.read_csv('data/sample_dataset.csv')
        except:
            df = self.generate_sample_data()
            df.to_csv('data/sample_dataset.csv', index=False)
        
        return df