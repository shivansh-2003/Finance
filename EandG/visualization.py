import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from utils import ColorPalette

class FinanceVisualizer:
    def __init__(self):
        self.colors = px.colors.qualitative.Plotly
    
    def create_category_pie_chart(self, category_data):
        """Create a pie chart showing expense distribution by category"""
        if not category_data:
            return None
            
        labels = list(category_data.keys())
        values = list(category_data.values())
        
        # Assign colors based on categories
        colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99']  # Example hardcoded colors
        
        # Debugging: Print the colors assigned
        print(f"Labels: {labels}")
        print(f"Values: {values}")
        print(f"Colors: {colors}")

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker=dict(colors=colors)  # Use the colors here
        )])
        
        fig.update_layout(
            title="Expenses by Category",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_monthly_trend_chart(self, data):
        """Create a line chart showing monthly spending trends"""
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["month"],
            y=df["total"],
            mode='lines+markers',
            name='Total Spent',
            line=dict(color=ColorPalette.primary, width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Monthly Spending Trend",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_category_trend_chart(self, category_trends, categories=None):
        """Create a line chart showing spending trends by category"""
        if not category_trends:
            return None
            
        fig = go.Figure()
        
        for category, data in category_trends.items():
            if categories and category not in categories:
                continue
                
            df = pd.DataFrame(data)
            color = ColorPalette.get_category_color(category)
            
            fig.add_trace(go.Scatter(
                x=df["month"],
                y=df["amount"],
                mode='lines+markers',
                name=category.title(),
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Category Spending Trends",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
