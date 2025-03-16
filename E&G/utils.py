import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import re

class UIHelper:
    @staticmethod
    def format_currency(amount):
        """Format a number as currency"""
        return f"${amount:,.2f}"

    @staticmethod
    def format_percentage(value):
        """Format a number as percentage"""
        return f"{value:.1f}%"
    
    @staticmethod
    def get_trend_color(value):
        """Get color based on trend direction"""
        if value > 0:
            return "red"
        elif value < 0:
            return "green"
        else:
            return "gray"
    
    @staticmethod
    def get_trend_arrow(value):
        """Get arrow based on trend direction"""
        if value > 0:
            return "↑"
        elif value < 0:
            return "↓"
        else:
            return "→"
    
    @staticmethod
    def create_metric_card(title, value, delta=None, delta_color=None, help_text=None):
        """Create a styled metric card"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"<p style='font-size:14px; color:gray;'>{title}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px; font-weight:bold;'>{value}</p>", unsafe_allow_html=True)
        
        if delta is not None:
            with col2:
                color = delta_color if delta_color else UIHelper.get_trend_color(delta)
                arrow = UIHelper.get_trend_arrow(delta)
                st.markdown(f"<p style='font-size:18px; color:{color}; text-align:right;'>{arrow} {abs(delta):.1f}%</p>", unsafe_allow_html=True)
        
        if help_text:
            st.caption(help_text)
    
    @staticmethod
    def create_progress_bar(title, current, target, percentage, color="#3498db"):
        """Create a styled progress bar"""
        st.markdown(f"<p style='font-size:14px; margin-bottom:5px;'>{title}</p>", unsafe_allow_html=True)
        
        progress_html = f"""
        <div style="width:100%; background-color:#f0f2f6; border-radius:10px; margin-bottom:5px;">
            <div style="width:{min(percentage, 100)}%; background-color:{color}; height:10px; border-radius:10px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px;">
            <span>{UIHelper.format_currency(current)}</span>
            <span>{UIHelper.format_percentage(percentage)}</span>
            <span>{UIHelper.format_currency(target)}</span>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    @staticmethod
    def display_insight_card(title, content, icon="💡"):
        """Display a card with insights"""
        st.markdown(
            f"""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:15px;">
                <h4 style="margin-top:0; display:flex; align-items:center;">
                    <span style="margin-right:8px;">{icon}</span>{title}
                </h4>
                <p style="margin-bottom:0;">{content}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

class DateHelper:
    @staticmethod
    def get_month_range(year, month):
        """Get start and end dates for a month"""
        start_date = datetime(year, month, 1)
        
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            
        return start_date, end_date
    
    @staticmethod
    def get_month_options():
        """Get list of months for selection"""
        now = datetime.now()
        options = []
        
        for i in range(12):
            month = now.month - i
            year = now.year
            
            while month <= 0:
                month += 12
                year -= 1
                
            date = datetime(year, month, 1)
            month_name = date.strftime("%B %Y")
            value = f"{year}-{month:02d}"
            
            options.append({"label": month_name, "value": value})
            
        return options
    
    @staticmethod
    def parse_month_value(value):
        """Parse month value into year and month"""
        match = re.match(r"(\d{4})-(\d{2})", value)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return year, month
        return None, None

class ColorPalette:
    """Color palette for consistent styling"""
    primary = "#3498db"
    secondary = "#2ecc71"
    accent = "#9b59b6"
    warning = "#f39c12"
    danger = "#e74c3c"
    
    category_colors = {
        "groceries": "#27ae60",
        "dining": "#e67e22",
        "entertainment": "#9b59b6",
        "utilities": "#3498db",
        "transport": "#f1c40f",
        "housing": "#e74c3c",
        "healthcare": "#1abc9c",
        "shopping": "#d35400",
        "education": "#8e44ad",
        "travel": "#16a085",
        "subscriptions": "#3498db",
        "personal": "#7f8c8d"
    }
    
    @classmethod
    def get_category_color(cls, category):
        """Get color for a specific category"""
        category = category.lower() if isinstance(category, str) else "other"
        return cls.category_colors.get(category, "#95a5a6")
