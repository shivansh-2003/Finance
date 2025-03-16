import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

class FinanceVisualizer:
    def __init__(self):
        self.colors = px.colors.qualitative.Plotly
    
    def create_category_pie_chart(self, category_amounts):
        """Create a pie chart of expenses by category"""
        if not category_amounts:
            return None
        
        labels = list(category_amounts.keys())
        values = list(category_amounts.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            insidetextorientation='radial',
            marker=dict(colors=self.colors[:len(labels)])
        )])
        
        fig.update_layout(
            title="Expenses by Category",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_monthly_trend_chart(self, monthly_data):
        """Create a line chart of spending trends over time"""
        if not monthly_data:
            return None
        
        df = pd.DataFrame(monthly_data)
        
        fig = px.line(
            df, 
            x="month", 
            y="total", 
            markers=True,
            title="Monthly Spending Trend",
            labels={"month": "Month", "total": "Total Spending ($)"}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_category_trend_chart(self, category_trends, categories=None):
        """Create a line chart of spending trends by category"""
        if not category_trends:
            return None
        
        # If categories is None, use all categories
        if categories is None:
            categories = list(category_trends.keys())
        else:
            # Filter to only include specified categories
            category_trends = {k: v for k, v in category_trends.items() if k in categories}
        
        # Prepare data for plotting
        all_data = []
        for category, data in category_trends.items():
            for point in data:
                all_data.append({
                    "month": point["month"],
                    "amount": point["amount"],
                    "category": category
                })
        
        df = pd.DataFrame(all_data)
        
        fig = px.line(
            df,
            x="month",
            y="amount",
            color="category",
            markers=True,
            title="Spending Trends by Category",
            labels={"month": "Month", "amount": "Amount ($)", "category": "Category"}
        )
        