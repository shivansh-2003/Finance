"""
ENHANCED PERSONAL FINANCE CURRICULUM CONTENT

This module provides detailed, practical content for each module in the curriculum,
including real-world examples, case studies, and interactive elements.
"""

# Example content for Money Basics module
MONEY_BASICS_CONTENT = {
    "key_concepts": {
        "money_functions": {
            "explanation": "Money serves as a medium of exchange, store of value, and unit of account.",
            "real_world_example": "When you use your debit card to buy groceries (medium of exchange), save money in a bank account (store of value), and track your monthly budget (unit of account).",
            "practical_exercise": "Track all your transactions for a week and categorize them by money function."
        },
        "income_types": {
            "explanation": "Income can be active (salary, wages) or passive (investments, rentals).",
            "real_world_example": "A software developer earning $80,000/year (active) and earning $200/month from a dividend stock portfolio (passive).",
            "practical_exercise": "List all your income sources and categorize them as active or passive."
        },
        "expense_management": {
            "explanation": "Understanding fixed vs. variable expenses and necessary vs. discretionary spending.",
            "real_world_example": "Fixed: $1,500 monthly rent; Variable: $200-400 monthly groceries; Necessary: utilities; Discretionary: streaming services.",
            "practical_exercise": "Create a spending diary and categorize each expense."
        }
    },
    "case_studies": [
        {
            "title": "Sarah's First Budget",
            "scenario": "Sarah, a recent graduate earning $45,000/year, needs to create her first budget.",
            "monthly_breakdown": {
                "take_home_pay": 3000,
                "fixed_expenses": {
                    "rent": 1200,
                    "utilities": 150,
                    "car_payment": 300,
                    "insurance": 200
                },
                "variable_expenses": {
                    "groceries": 400,
                    "gas": 150,
                    "entertainment": 200
                },
                "savings": 400
            },
            "learning_points": [
                "Importance of tracking all income and expenses",
                "Building an emergency fund with leftover money",
                "Identifying areas for potential savings"
            ]
        }
    ],
    "interactive_elements": {
        "budget_calculator": {
            "description": "Interactive tool to create a personal budget",
            "inputs": ["monthly_income", "fixed_expenses", "variable_expenses"],
            "outputs": ["disposable_income", "savings_potential", "expense_ratio"]
        },
        "expense_tracker": {
            "description": "Daily expense tracking template",
            "categories": ["housing", "transportation", "food", "utilities", "entertainment", "savings"],
            "features": ["category_totals", "spending_trends", "budget_alerts"]
        }
    },
    "assessment_questions": [
        {
            "question": "Which of the following is an example of a fixed expense?",
            "options": [
                "Monthly rent payment",
                "Grocery shopping",
                "Entertainment expenses",
                "Gas for your car"
            ],
            "correct_answer": "Monthly rent payment",
            "explanation": "Fixed expenses remain constant each month, like rent, loan payments, or insurance premiums."
        }
    ]
}

# Example content for Budgeting module
BUDGETING_CONTENT = {
    "key_concepts": {
        "50_30_20_rule": {
            "explanation": "Allocate 50% to needs, 30% to wants, and 20% to savings/debt repayment.",
            "real_world_example": "On a $4,000 monthly income: $2,000 for needs (rent, utilities, groceries), $1,200 for wants (dining out, entertainment), $800 for savings/debt.",
            "practical_exercise": "Calculate your current spending ratios and compare to the 50/30/20 rule."
        },
        "zero_based_budgeting": {
            "explanation": "Assign every dollar a specific purpose until your income minus expenses equals zero.",
            "real_world_example": "Monthly income $3,000: Rent $1,200, Utilities $200, Groceries $400, Transportation $300, Entertainment $200, Savings $500, Emergency Fund $200.",
            "practical_exercise": "Create a zero-based budget for your next month."
        }
    },
    "case_studies": [
        {
            "title": "The Debt-Free Journey",
            "scenario": "Michael has $10,000 in credit card debt and wants to become debt-free while building savings.",
            "monthly_plan": {
                "income": 4500,
                "essential_expenses": 2500,
                "debt_payment": 1000,
                "emergency_fund": 500,
                "discretionary": 500
            },
            "timeline": "12 months",
            "outcome": "Debt reduced to $2,000, $6,000 emergency fund built"
        }
    ]
}

# Example content for Investment module
INVESTMENT_CONTENT = {
    "key_concepts": {
        "compound_interest": {
            "explanation": "Earning interest on both principal and previously earned interest.",
            "real_world_example": "$10,000 invested at 7% annual return becomes $19,672 after 10 years through compound interest.",
            "practical_exercise": "Calculate compound interest for different investment scenarios using the provided calculator."
        },
        "diversification": {
            "explanation": "Spreading investments across different asset types to manage risk.",
            "real_world_example": "A balanced portfolio with 60% stocks (S&P 500 index fund), 30% bonds (government/corporate), and 10% cash/alternatives.",
            "practical_exercise": "Analyze your current investments and calculate asset allocation percentages."
        }
    },
    "case_studies": [
        {
            "title": "Building a Retirement Portfolio",
            "scenario": "Emma, age 25, starts investing for retirement with $500 monthly contributions.",
            "investment_strategy": {
                "initial_allocation": {
                    "stocks": "90%",
                    "bonds": "10%"
                },
                "adjustment": "Decrease stock allocation by 1% per year after age 35",
                "target_retirement": "Age 65",
                "projected_outcome": "$1.2M (assuming 7% average annual return)"
            }
        }
    ]
}

def get_module_content(module_id: str) -> dict:
    """Get the enhanced content for a specific module."""
    content_map = {
        "module_1_1": MONEY_BASICS_CONTENT,
        "module_1_2": BUDGETING_CONTENT,
        "module_3_1": INVESTMENT_CONTENT,
        # Add more modules as they're developed
    }
    return content_map.get(module_id, {}) 