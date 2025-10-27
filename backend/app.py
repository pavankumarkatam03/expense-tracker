from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import random
import os
import csv
from io import StringIO, BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
CORS(app)

print("🚀 Personal Finance Tracker Backend Starting...")

# Sample categories - will be updated from settings
INCOME_CATEGORIES = ['Salary', 'Freelance', 'Investments', 'Gifts', 'Other Income']
EXPENSE_CATEGORIES = ['Food', 'Rent', 'Transportation', 'Utilities', 'Entertainment', 'Healthcare', 'Education', 'Shopping', 'Other Expenses']

class FinanceTracker:
    def __init__(self):
        self.transactions = []
        self.budgets = []
        self.settings = {
            'income_categories': INCOME_CATEGORIES.copy(),
            'expense_categories': EXPENSE_CATEGORIES.copy(),
            'currency': '₹',
            'dark_mode': False
        }
        self.load_sample_data()
    
    def load_sample_data(self):
        """Load sample data if no transactions exist"""
        if not self.transactions:
            print("📊 Loading sample data...")
            sample_data = self.generate_sample_data()
            for transaction in sample_data:
                self.add_transaction(transaction)
            
            # Add sample budgets
            self.budgets = [
                {'id': 1, 'category': 'Food', 'amount': 8000},
                {'id': 2, 'category': 'Rent', 'amount': 15000},
                {'id': 3, 'category': 'Transportation', 'amount': 3000},
                {'id': 4, 'category': 'Entertainment', 'amount': 4000},
                {'id': 5, 'category': 'Utilities', 'amount': 2000}
            ]
            print("✅ Sample data loaded successfully!")
    
    def add_transaction(self, transaction):
        # Ensure amount is stored as float
        transaction['amount'] = float(transaction['amount'])
        transaction['id'] = len(self.transactions) + 1
        transaction['created_at'] = datetime.now().isoformat()
        
        # Handle recurring transactions
        if transaction.get('recurring'):
            self._create_recurring_transactions(transaction)
        else:
            self.transactions.append(transaction)
        
        return transaction
    
    def _create_recurring_transactions(self, base_transaction):
        """Create recurring transactions based on frequency"""
        frequency = base_transaction['recurring']['frequency']
        end_date = base_transaction['recurring'].get('end_date')
        iterations = base_transaction['recurring'].get('iterations')
        
        base_date = datetime.strptime(base_transaction['date'], '%Y-%m-%d')
        current_date = base_date
        
        count = 0
        max_iterations = iterations if iterations else 12  # Default to 12 months
        
        while count < max_iterations:
            if end_date and current_date > datetime.strptime(end_date, '%Y-%m-%d'):
                break
                
            # Create transaction for this occurrence
            recurring_transaction = base_transaction.copy()
            recurring_transaction['date'] = current_date.strftime('%Y-%m-%d')
            recurring_transaction['id'] = len(self.transactions) + 1
            recurring_transaction['created_at'] = datetime.now().isoformat()
            recurring_transaction['is_recurring'] = True
            recurring_transaction['recurring_parent_id'] = base_transaction['id']
            
            self.transactions.append(recurring_transaction)
            
            # Move to next date based on frequency
            if frequency == 'weekly':
                current_date += timedelta(days=7)
            elif frequency == 'bi-weekly':
                current_date += timedelta(days=14)
            elif frequency == 'monthly':
                current_date = current_date.replace(month=current_date.month + 1)
            elif frequency == 'quarterly':
                current_date = current_date.replace(month=current_date.month + 3)
            elif frequency == 'yearly':
                current_date = current_date.replace(year=current_date.year + 1)
            
            count += 1
    
    def get_transactions(self, filters=None):
        filtered = self.transactions.copy()
        
        if filters:
            if filters.get('type') and filters['type'] != 'all':
                filtered = [t for t in filtered if t['type'] == filters['type']]
            
            if filters.get('category') and filters['category'] != 'all':
                filtered = [t for t in filtered if t['category'] == filters['category']]
            
            if filters.get('month') and filters.get('year'):
                filtered = [
                    t for t in filtered 
                    if self.parse_date(t['date']).month == filters['month'] and
                    self.parse_date(t['date']).year == filters['year']
                ]
        
        return sorted(filtered, key=lambda x: self.parse_date(x['date']), reverse=True)
    
    def get_summary(self, month=None, year=None):
        if month is None:
            month = datetime.now().month
        if year is None:
            year = datetime.now().year
        
        monthly_transactions = [
            t for t in self.transactions 
            if self.parse_date(t['date']).month == month and
            self.parse_date(t['date']).year == year
        ]
        
        # Convert amounts to float to ensure they're numbers
        income = sum(float(t['amount']) for t in monthly_transactions if t['type'] == 'income')
        expenses = sum(float(t['amount']) for t in monthly_transactions if t['type'] == 'expense')
        balance = income - expenses
        
        # Recent transactions (last 5)
        recent = sorted(self.transactions, key=lambda x: self.parse_date(x['date']), reverse=True)[:5]
        
        # Category breakdown - ensure "Other Expenses" is last
        category_breakdown = {}
        for transaction in monthly_transactions:
            if transaction['type'] == 'expense':
                category = transaction['category']
                amount = float(transaction['amount'])
                category_breakdown[category] = category_breakdown.get(category, 0) + amount
        
        # Sort categories, putting "Other Expenses" last
        sorted_categories = sorted(
            category_breakdown.items(),
            key=lambda x: (x[0] == "Other Expenses", x[0])
        )
        category_breakdown = dict(sorted_categories)
        
        return {
            'income': income,
            'expenses': expenses,
            'balance': balance,
            'recent_transactions': recent,
            'category_breakdown': category_breakdown
        }
    
    def get_budget_summary(self, month=None, year=None):
        if month is None:
            month = datetime.now().month
        if year is None:
            year = datetime.now().year
        
        monthly_expenses = [
            t for t in self.transactions 
            if t['type'] == 'expense' and
            self.parse_date(t['date']).month == month and
            self.parse_date(t['date']).year == year
        ]
        
        # Remove duplicate budgets by category
        unique_budgets = {}
        for budget in self.budgets:
            unique_budgets[budget['category']] = budget
        
        budget_summary = []
        for category, budget in unique_budgets.items():
            spent = sum(float(t['amount']) for t in monthly_expenses if t['category'] == category)
            budget_summary.append({
                'id': budget['id'],
                'category': category,
                'budget': float(budget['amount']),
                'spent': spent,
                'remaining': float(budget['amount']) - spent
            })
        
        return budget_summary
    
    def get_monthly_reports(self):
        """Get monthly summaries for all months with data"""
        if not self.transactions:
            return []
        
        # Get unique month-year combinations
        month_years = set()
        for transaction in self.transactions:
            date = self.parse_date(transaction['date'])
            month_years.add((date.year, date.month))
        
        reports = []
        for year, month in month_years:
            monthly_data = self.get_summary(month, year)
            
            # Fix for infinity savings rate - check if income is 0
            savings_rate = 0
            if monthly_data['income'] > 0:
                savings_rate = (monthly_data['balance'] / monthly_data['income']) * 100
            
            reports.append({
                'year': year,
                'month': month,
                'month_name': datetime(year, month, 1).strftime('%B %Y'),
                'income': monthly_data['income'],
                'expenses': monthly_data['expenses'],
                'savings': monthly_data['balance'],
                'savings_rate': savings_rate,  # Fixed: No more infinity
                'transaction_count': len([t for t in self.transactions 
                                        if self.parse_date(t['date']).month == month and
                                        self.parse_date(t['date']).year == year])
            })
        
        return sorted(reports, key=lambda x: (x['year'], x['month']), reverse=True)
    
    def parse_date(self, date_str):
        return datetime.fromisoformat(date_str) if 'T' in date_str else datetime.strptime(date_str, '%Y-%m-%d')
    
    def generate_sample_data(self):
        """Generate comprehensive sample data"""
        transactions = []
        current_date = datetime.now()
        
        # Income transactions (last 3 months)
        for i in range(3):
            date = current_date - timedelta(days=30*i)
            transactions.append({
                'type': 'income',
                'date': date.strftime('%Y-%m-%d'),
                'amount': 60000 - (i * 5000),
                'category': 'Salary',
                'description': f'Monthly salary {date.strftime("%B")}'
            })
        
        # Expense transactions
        expense_patterns = {
            'Food': (200, 2000, 15),
            'Rent': (12000, 15000, 1),
            'Transportation': (50, 1000, 8),
            'Utilities': (500, 2000, 3),
            'Entertainment': (200, 1500, 6),
            'Healthcare': (100, 1000, 2),
            'Education': (500, 3000, 2),
            'Shopping': (300, 2500, 4),
            'Other Expenses': (100, 800, 3)
        }
        
        for category, (min_amt, max_amt, freq) in expense_patterns.items():
            for i in range(freq * 3):
                amount = random.randint(min_amt, max_amt)
                days_ago = random.randint(1, 90)
                transactions.append({
                    'type': 'expense',
                    'date': (current_date - timedelta(days=days_ago)).strftime('%Y-%m-%d'),
                    'amount': amount,
                    'category': category,
                    'description': f'{category} purchase'
                })
        
        return transactions

# Initialize tracker
tracker = FinanceTracker()

# API Routes
@app.route('/api/transactions', methods=['GET', 'POST'])
def handle_transactions():
    if request.method == 'GET':
        filters = request.args.to_dict()
        transactions = tracker.get_transactions(filters)
        return jsonify(transactions)
    else:
        transaction = request.json
        # Ensure amount is converted to float
        if 'amount' in transaction:
            try:
                transaction['amount'] = float(transaction['amount'])
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid amount format'}), 400
        result = tracker.add_transaction(transaction)
        return jsonify(result)

@app.route('/api/transactions/<int:transaction_id>', methods=['PUT', 'DELETE'])
def handle_single_transaction(transaction_id):
    if request.method == 'PUT':
        updated_transaction = request.json
        # Ensure amount is converted to float
        if 'amount' in updated_transaction:
            try:
                updated_transaction['amount'] = float(updated_transaction['amount'])
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid amount format'}), 400
                
        for i, transaction in enumerate(tracker.transactions):
            if transaction['id'] == transaction_id:
                tracker.transactions[i] = {**transaction, **updated_transaction}
                return jsonify(tracker.transactions[i])
        return jsonify({'error': 'Transaction not found'}), 404
    else:
        tracker.transactions = [t for t in tracker.transactions if t['id'] != transaction_id]
        return jsonify({'message': 'Transaction deleted'})

@app.route('/api/budgets', methods=['GET', 'POST', 'PUT'])
def handle_budgets():
    if request.method == 'GET':
        return jsonify(tracker.budgets)
    elif request.method == 'POST':
        budget = request.json
        # Check if budget already exists for this category
        existing_budget = next((b for b in tracker.budgets if b['category'] == budget['category']), None)
        if existing_budget:
            return jsonify({'error': 'Budget already exists for this category'}), 400
        
        # Ensure amount is converted to float
        if 'amount' in budget:
            try:
                budget['amount'] = float(budget['amount'])
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid amount format'}), 400
        budget['id'] = len(tracker.budgets) + 1
        tracker.budgets.append(budget)
        return jsonify(budget)
    else:  # PUT
        updated_budget = request.json
        # Ensure amount is converted to float
        if 'amount' in updated_budget:
            try:
                updated_budget['amount'] = float(updated_budget['amount'])
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid amount format'}), 400
                
        for i, budget in enumerate(tracker.budgets):
            if budget['id'] == updated_budget['id']:
                tracker.budgets[i] = {**budget, **updated_budget}
                return jsonify(tracker.budgets[i])
        return jsonify({'error': 'Budget not found'}), 404

@app.route('/api/budgets/<int:budget_id>', methods=['DELETE'])
def delete_budget(budget_id):
    tracker.budgets = [b for b in tracker.budgets if b['id'] != budget_id]
    return jsonify({'message': 'Budget deleted'})

@app.route('/api/dashboard/summary')
def get_dashboard_summary():
    summary = tracker.get_summary()
    return jsonify(summary)

@app.route('/api/budget/summary')
def get_budget_summary():
    summary = tracker.get_budget_summary()
    return jsonify(summary)

@app.route('/api/reports/monthly')
def get_monthly_reports():
    reports = tracker.get_monthly_reports()
    return jsonify(reports)

@app.route('/api/predict-future', methods=['POST'])
def predict_future_expenses():
    try:
        data = request.json
        months_ahead = data.get('months_ahead', 6)
        
        predictions = []
        current_date = datetime.now()
        
        base_patterns = {
            'Food': 6000,
            'Rent': 15000,
            'Transportation': 2500,
            'Utilities': 1500,
            'Entertainment': 3000,
            'Healthcare': 1200,
            'Education': 2000,
            'Shopping': 3500,
            'Other Expenses': 1000
        }
        
        for i in range(1, months_ahead + 1):
            future_date = current_date + timedelta(days=30 * i)
            monthly_predictions = {}
            
            for category, base in base_patterns.items():
                seasonal = get_seasonal_factor(category, future_date.month)
                variation = 0.9 + (0.2 * random.random())
                predicted = base * seasonal * variation
                monthly_predictions[category] = round(predicted, 2)
            
            predictions.append({
                'month': future_date.strftime('%Y-%m'),
                'predictions': monthly_predictions,
                'total_predicted': round(sum(monthly_predictions.values()), 2)
            })
        
        return jsonify({
            'predictions': predictions,
            'message': f'Generated predictions for {months_ahead} months'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_seasonal_factor(category, month):
    factors = {
        'Food': {12: 1.2, 1: 1.1, 6: 0.9, 7: 0.9},
        'Transportation': {12: 1.3, 1: 1.2, 6: 1.4, 7: 1.4},
        'Entertainment': {12: 1.4, 1: 1.3, 6: 1.2, 7: 1.2},
        'Shopping': {12: 1.5, 11: 1.3, 6: 1.1, 1: 0.7}
    }
    return factors.get(category, {}).get(month, 1.0)

@app.route('/api/categories/income')
def get_income_categories():
    return jsonify(tracker.settings['income_categories'])

@app.route('/api/categories/expense')
def get_expense_categories():
    return jsonify(tracker.settings['expense_categories'])

@app.route('/api/settings', methods=['GET', 'PUT'])
def handle_settings():
    if request.method == 'GET':
        return jsonify(tracker.settings)
    else:
        new_settings = request.json
        # Update categories if provided
        if 'income_categories' in new_settings:
            tracker.settings['income_categories'] = new_settings['income_categories']
        if 'expense_categories' in new_settings:
            tracker.settings['expense_categories'] = new_settings['expense_categories']
        if 'currency' in new_settings:
            tracker.settings['currency'] = new_settings['currency']
        if 'dark_mode' in new_settings:
            tracker.settings['dark_mode'] = new_settings['dark_mode']
        return jsonify(tracker.settings)

@app.route('/api/settings/categories', methods=['POST', 'DELETE'])
def manage_categories():
    if request.method == 'POST':
        data = request.json
        category_type = data.get('type')  # 'income' or 'expense'
        category_name = data.get('name')
        
        if not category_type or not category_name:
            return jsonify({'error': 'Type and name are required'}), 400
        
        if category_type == 'income':
            if category_name not in tracker.settings['income_categories']:
                tracker.settings['income_categories'].append(category_name)
        elif category_type == 'expense':
            if category_name not in tracker.settings['expense_categories']:
                tracker.settings['expense_categories'].append(category_name)
        else:
            return jsonify({'error': 'Invalid category type'}), 400
        
        return jsonify({'message': 'Category added successfully'})
    
    else:  # DELETE
        data = request.json
        category_type = data.get('type')
        category_name = data.get('name')
        
        if not category_type or not category_name:
            return jsonify({'error': 'Type and name are required'}), 400
        
        if category_type == 'income':
            if category_name in tracker.settings['income_categories']:
                tracker.settings['income_categories'].remove(category_name)
        elif category_type == 'expense':
            if category_name in tracker.settings['expense_categories']:
                tracker.settings['expense_categories'].remove(category_name)
        else:
            return jsonify({'error': 'Invalid category type'}), 400
        
        return jsonify({'message': 'Category deleted successfully'})

def generate_pdf(data, title, columns, filename):
    """Generate PDF file from data"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    # Add title
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_paragraph = Paragraph(title, title_style)
    elements.append(title_paragraph)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add generation date
    date_style = styles['Normal']
    date_paragraph = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style)
    elements.append(date_paragraph)
    elements.append(Spacer(1, 0.25*inch))
    
    # Prepare table data
    table_data = [columns]
    
    for item in data:
        row = []
        for col in columns:
            if col in item:
                value = item[col]
                if isinstance(value, (int, float)) and col.lower() in ['amount', 'budget', 'spent', 'remaining', 'income', 'expenses', 'savings']:
                    row.append(f"₹{value:,.2f}")
                else:
                    row.append(str(value))
            else:
                row.append('')
        table_data.append(row)
    
    # Create table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    doc.build(elements)
    
    buffer.seek(0)
    return buffer

@app.route('/api/export/<export_type>', methods=['GET'])
def export_data(export_type):
    try:
        format_type = request.args.get('format', 'csv')
        
        if export_type == 'transactions':
            transactions = tracker.get_transactions()
            
            if format_type == 'pdf':
                columns = ['ID', 'Date', 'Type', 'Category', 'Amount', 'Description']
                data = []
                for t in transactions:
                    data.append({
                        'ID': t['id'],
                        'Date': t['date'],
                        'Type': t['type'].title(),
                        'Category': t['category'],
                        'Amount': f"₹{t['amount']:,.2f}",
                        'Description': t.get('description', 'N/A')
                    })
                
                pdf_buffer = generate_pdf(data, 'Transactions Export', columns, 'transactions_export.pdf')
                return app.response_class(
                    pdf_buffer.getvalue(),
                    mimetype='application/pdf',
                    headers={'Content-Disposition': 'attachment; filename=transactions_export.pdf'}
                )
                
            elif format_type == 'excel':
                # Create Excel file
                df_data = []
                for t in transactions:
                    df_data.append({
                        'ID': t['id'],
                        'Date': t['date'],
                        'Type': t['type'].title(),
                        'Category': t['category'],
                        'Amount': t['amount'],
                        'Description': t.get('description', ''),
                        'Created At': t['created_at']
                    })
                
                df = pd.DataFrame(df_data)
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Transactions', index=False)
                excel_buffer.seek(0)
                
                return app.response_class(
                    excel_buffer.getvalue(),
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={'Content-Disposition': 'attachment; filename=transactions_export.xlsx'}
                )
                
            else:  # CSV
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(['ID', 'Date', 'Type', 'Category', 'Amount', 'Description', 'Created At'])
                
                for transaction in transactions:
                    writer.writerow([
                        transaction['id'],
                        transaction['date'],
                        transaction['type'],
                        transaction['category'],
                        f"₹{transaction['amount']}",
                        transaction.get('description', ''),
                        transaction['created_at']
                    ])
                
                csv_data = output.getvalue()
                output.close()
                
                return app.response_class(
                    response=csv_data,
                    status=200,
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=transactions_export.csv'}
                )
            
        elif export_type == 'budgets':
            budgets = tracker.budgets
            
            if format_type == 'pdf':
                columns = ['ID', 'Category', 'Budget Amount']
                data = []
                for b in budgets:
                    data.append({
                        'ID': b['id'],
                        'Category': b['category'],
                        'Budget Amount': f"₹{b['amount']:,.2f}"
                    })
                
                pdf_buffer = generate_pdf(data, 'Budgets Export', columns, 'budgets_export.pdf')
                return app.response_class(
                    pdf_buffer.getvalue(),
                    mimetype='application/pdf',
                    headers={'Content-Disposition': 'attachment; filename=budgets_export.pdf'}
                )
                
            elif format_type == 'excel':
                df_data = []
                for b in budgets:
                    df_data.append({
                        'ID': b['id'],
                        'Category': b['category'],
                        'Budget Amount': b['amount'],
                        'Created At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                df = pd.DataFrame(df_data)
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Budgets', index=False)
                excel_buffer.seek(0)
                
                return app.response_class(
                    excel_buffer.getvalue(),
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={'Content-Disposition': 'attachment; filename=budgets_export.xlsx'}
                )
                
            else:  # CSV
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(['ID', 'Category', 'Budget Amount', 'Created At'])
                
                for budget in budgets:
                    writer.writerow([
                        budget['id'],
                        budget['category'],
                        f"₹{budget['amount']}",
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ])
                
                csv_data = output.getvalue()
                output.close()
                
                return app.response_class(
                    response=csv_data,
                    status=200,
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=budgets_export.csv'}
                )
            
        elif export_type == 'reports':
            reports = tracker.get_monthly_reports()
            
            if format_type == 'pdf':
                columns = ['Month', 'Income', 'Expenses', 'Savings', 'Savings Rate (%)', 'Transaction Count']
                data = []
                for r in reports:
                    data.append({
                        'Month': r['month_name'],
                        'Income': f"₹{r['income']:,.2f}",
                        'Expenses': f"₹{r['expenses']:,.2f}",
                        'Savings': f"₹{r['savings']:,.2f}",
                        'Savings Rate (%)': f"{r['savings_rate']:.1f}%",
                        'Transaction Count': r['transaction_count']
                    })
                
                pdf_buffer = generate_pdf(data, 'Financial Reports Export', columns, 'financial_reports_export.pdf')
                return app.response_class(
                    pdf_buffer.getvalue(),
                    mimetype='application/pdf',
                    headers={'Content-Disposition': 'attachment; filename=financial_reports_export.pdf'}
                )
                
            elif format_type == 'excel':
                df_data = []
                for r in reports:
                    df_data.append({
                        'Month': r['month_name'],
                        'Income': r['income'],
                        'Expenses': r['expenses'],
                        'Savings': r['savings'],
                        'Savings Rate (%)': r['savings_rate'],
                        'Transaction Count': r['transaction_count']
                    })
                
                df = pd.DataFrame(df_data)
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Financial Reports', index=False)
                excel_buffer.seek(0)
                
                return app.response_class(
                    excel_buffer.getvalue(),
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={'Content-Disposition': 'attachment; filename=financial_reports_export.xlsx'}
                )
                
            else:  # CSV
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(['Month', 'Income', 'Expenses', 'Savings', 'Savings Rate (%)', 'Transaction Count'])
                
                for report in reports:
                    writer.writerow([
                        report['month_name'],
                        f"₹{report['income']}",
                        f"₹{report['expenses']}",
                        f"₹{report['savings']}",
                        f"{report['savings_rate']:.1f}%",
                        report['transaction_count']
                    ])
                
                csv_data = output.getvalue()
                output.close()
                
                return app.response_class(
                    response=csv_data,
                    status=200,
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=financial_reports_export.csv'}
                )
            
        elif export_type == 'predictions':
            predictions_response = predict_future_expenses()
            predictions_data = predictions_response.get_json()
            predictions = predictions_data.get('predictions', [])
            
            if format_type == 'pdf':
                columns = ['Month', 'Total Predicted', 'Food', 'Rent', 'Transportation', 'Utilities', 'Entertainment', 'Healthcare', 'Education', 'Shopping', 'Other Expenses']
                data = []
                for p in predictions:
                    data.append({
                        'Month': p['month'],
                        'Total Predicted': f"₹{p['total_predicted']:,.2f}",
                        'Food': f"₹{p['predictions'].get('Food', 0):,.2f}",
                        'Rent': f"₹{p['predictions'].get('Rent', 0):,.2f}",
                        'Transportation': f"₹{p['predictions'].get('Transportation', 0):,.2f}",
                        'Utilities': f"₹{p['predictions'].get('Utilities', 0):,.2f}",
                        'Entertainment': f"₹{p['predictions'].get('Entertainment', 0):,.2f}",
                        'Healthcare': f"₹{p['predictions'].get('Healthcare', 0):,.2f}",
                        'Education': f"₹{p['predictions'].get('Education', 0):,.2f}",
                        'Shopping': f"₹{p['predictions'].get('Shopping', 0):,.2f}",
                        'Other Expenses': f"₹{p['predictions'].get('Other Expenses', 0):,.2f}"
                    })
                
                pdf_buffer = generate_pdf(data, 'Future Predictions Export', columns, 'future_predictions_export.pdf')
                return app.response_class(
                    pdf_buffer.getvalue(),
                    mimetype='application/pdf',
                    headers={'Content-Disposition': 'attachment; filename=future_predictions_export.pdf'}
                )
                
            elif format_type == 'excel':
                df_data = []
                for p in predictions:
                    row = {
                        'Month': p['month'],
                        'Total Predicted': p['total_predicted']
                    }
                    # Add each category
                    for category, amount in p['predictions'].items():
                        row[category] = amount
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Future Predictions', index=False)
                excel_buffer.seek(0)
                
                return app.response_class(
                    excel_buffer.getvalue(),
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={'Content-Disposition': 'attachment; filename=future_predictions_export.xlsx'}
                )
                
            else:  # CSV
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(['Month', 'Total Predicted', 'Food', 'Rent', 'Transportation', 'Utilities', 'Entertainment', 'Healthcare', 'Education', 'Shopping', 'Other Expenses'])
                
                for prediction in predictions:
                    writer.writerow([
                        prediction['month'],
                        f"₹{prediction['total_predicted']}",
                        f"₹{prediction['predictions'].get('Food', 0)}",
                        f"₹{prediction['predictions'].get('Rent', 0)}",
                        f"₹{prediction['predictions'].get('Transportation', 0)}",
                        f"₹{prediction['predictions'].get('Utilities', 0)}",
                        f"₹{prediction['predictions'].get('Entertainment', 0)}",
                        f"₹{prediction['predictions'].get('Healthcare', 0)}",
                        f"₹{prediction['predictions'].get('Education', 0)}",
                        f"₹{prediction['predictions'].get('Shopping', 0)}",
                        f"₹{prediction['predictions'].get('Other Expenses', 0)}"
                    ])
                
                csv_data = output.getvalue()
                output.close()
                
                return app.response_class(
                    response=csv_data,
                    status=200,
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=future_predictions_export.csv'}
                )
            
        else:
            return jsonify({'error': 'Invalid export type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        'message': 'Personal Finance Tracker API',
        'status': 'running',
        'endpoints': {
            'dashboard': '/api/dashboard/summary',
            'transactions': '/api/transactions',
            'budgets': '/api/budgets',
            'reports': '/api/reports/monthly',
            'predictions': '/api/predict-future',
            'settings': '/api/settings',
            'export': '/api/export/<type>?format=csv|pdf|excel'
        }
    })

@app.route('/api/reset-data', methods=['POST'])
def reset_data():
    """Reset all data and reload sample data"""
    global tracker
    tracker = FinanceTracker()
    return jsonify({'message': 'Data reset successfully'})
@app.route('/api/analytics/data')
def get_analytics_data():
    """Get analytics data for charts and trends"""
    try:
        reports = tracker.get_monthly_reports()
        
        if not reports:
            return jsonify({
                'income_vs_expenses': {
                    'months': [],
                    'income': [],
                    'expenses': [],
                    'savings': []
                },
                'category_trends': {},
                'recent_months': []
            })
        
        # Get last 6 months for trends
        recent_reports = reports[:6][::-1]  # Reverse to show oldest first
        
        # Income vs Expenses data
        months = [report['month_name'] for report in recent_reports]
        income_data = [report['income'] for report in recent_reports]
        expenses_data = [report['expenses'] for report in recent_reports]
        savings_data = [report['savings'] for report in recent_reports]
        
        # Category trends (last 6 months)
        category_trends = {}
        recent_months = months
        
        # Get expense categories from recent transactions
        all_categories = set()
        for transaction in tracker.transactions:
            if transaction['type'] == 'expense':
                all_categories.add(transaction['category'])
        
        # Calculate monthly totals for each category
        for category in all_categories:
            category_totals = []
            for report in recent_reports:
                # Get transactions for this month and category
                month_transactions = [
                    t for t in tracker.transactions 
                    if tracker.parse_date(t['date']).month == report['month'] and
                    tracker.parse_date(t['date']).year == report['year'] and
                    t['type'] == 'expense' and
                    t['category'] == category
                ]
                total = sum(t['amount'] for t in month_transactions)
                category_totals.append(total)
            category_trends[category] = category_totals
        
        return jsonify({
            'income_vs_expenses': {
                'months': months,
                'income': income_data,
                'expenses': expenses_data,
                'savings': savings_data
            },
            'category_trends': category_trends,
            'recent_months': recent_months
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("✅ Backend initialized successfully!")
    print("📍 API Server running on: http://localhost:5000")
    print("📊 Available endpoints:")
    print("   GET  /api/dashboard/summary")
    print("   GET  /api/transactions")
    print("   POST /api/transactions") 
    print("   GET  /api/budgets")
    print("   GET  /api/reports/monthly")
    print("   POST /api/predict-future")
    print("   GET/PUT /api/settings")
    print("   GET  /api/export/<type>?format=csv|pdf|excel")
    print("   POST /api/reset-data (reset all data)")
    print("\n🚀 Starting server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
    