# finance_agent/personal_finance.py

from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool

class PersonalFinanceAdvisor:
    """
    Agent for providing personalized financial advice based on individual situations.
    """
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        # Set up tools for different financial calculations
        self.tools = [
            Tool(
                name="BudgetOptimizer",
                func=self._optimize_budget,
                description="Optimizes budget allocation based on income, expenses, and financial goals"
            ),
            Tool(
                name="InvestmentRecommender",
                func=self._recommend_investments,
                description="Recommends investment allocations based on goals, risk tolerance, and time horizon"
            ),
            Tool(
                name="DebtRepaymentPlanner",
                func=self._plan_debt_repayment,
                description="Creates a debt repayment plan prioritizing high-interest debt"
            ),
            Tool(
                name="SavingsCalculator",
                func=self._calculate_savings_plan,
                description="Calculates savings plan to reach specific financial goals"
            ),
            Tool(
                name="TaxOptimizer",
                func=self._optimize_tax_strategy,
                description="Suggests tax optimization strategies based on income and investment situation"
            )
        ]
        
        # Create the system prompt for the financial advisor
        system_prompt = """You are an expert personal financial advisor with years of experience helping individuals manage their finances effectively.

Your expertise includes:
- Budgeting and expense optimization
- Debt management and reduction strategies
- Investment planning and portfolio allocation
- Retirement planning
- Tax optimization
- Emergency fund planning
- Financial goal setting and achievement

When providing advice:
1. Always consider the person's complete financial situation
2. Prioritize debt repayment, especially high-interest debt
3. Emphasize the importance of emergency funds
4. Consider tax implications of recommendations
5. Provide specific, actionable steps they can take
6. Explain the reasoning behind your recommendations
7. Consider both short-term needs and long-term goals
8. Be sensitive to different risk tolerances
9. Acknowledge that personal finance is personal - there's no one-size-fits-all approach

Use the available tools to calculate optimal strategies before providing your final recommendations.
"""
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def provide_advice(self, situation, income, expenses, savings, debt=None, goals=None, risk_tolerance=None, time_horizon=None):
        """
        Provide personalized financial advice based on the provided information.
        
        Parameters:
        - situation: Description of the current financial situation
        - income: Monthly or annual income
        - expenses: Dictionary of expense categories and amounts
        - savings: Current savings amount
        - debt: Optional dictionary of debt accounts and amounts
        - goals: Optional list of financial goals
        - risk_tolerance: Optional risk tolerance level (conservative, moderate, aggressive)
        - time_horizon: Optional investment time horizon (short, medium, long)
        
        Returns:
        - Dictionary containing the financial advice
        """
        # Create a structured input for the agent
        structured_input = f"""
        Financial Situation: {situation}
        
        Financial Details:
        - Monthly Income: ${income}
        - Current Savings: ${savings}
        """
        
        # Add expenses details
        if expenses:
            structured_input += "- Monthly Expenses:\n"
            for category, amount in expenses.items():
                structured_input += f"  * {category}: ${amount}\n"
        
        # Add debt details if provided
        if debt:
            structured_input += "- Current Debts:\n"
            for account, amount in debt.items():
                structured_input += f"  * {account}: ${amount}\n"
        
        # Add financial goals if provided
        if goals:
            structured_input += "- Financial Goals:\n"
            for goal in goals:
                structured_input += f"  * {goal}\n"
        
        # Add risk tolerance if provided
        if risk_tolerance:
            structured_input += f"- Risk Tolerance: {risk_tolerance}\n"
        
        # Add time horizon if provided
        if time_horizon:
            structured_input += f"- Investment Time Horizon: {time_horizon}\n"
        
        structured_input += """
        Based on my financial situation, please provide comprehensive advice including:
        1. Budget optimization recommendations
        2. Debt management strategy (if applicable)
        3. Savings plan to reach my goals
        4. Investment recommendations appropriate for my risk tolerance and time horizon
        5. Tax optimization suggestions
        6. Any other relevant financial advice for my situation
        
        Please be specific and provide actionable steps I can take to improve my financial health.
        """
        
        # Run the agent
        response = self.agent_executor.invoke({"input": structured_input})
        
        # Return the response
        return {"response": response["output"]}
    
    def process_natural_language(self, query):
        """
        Process a natural language query about personal finance.
        
        Parameters:
        - query: The natural language query or description of financial situation
        
        Returns:
        - String containing financial advice
        """
        response = self.agent_executor.invoke({"input": query})
        return response["output"]
    
    def _optimize_budget(self, income: float, expenses: Dict[str, float], savings_target: Optional[float] = None) -> Dict[str, Any]:
        """Optimize budget allocation based on income and expenses"""
        # Calculate total expenses
        total_expenses = sum(expenses.values())
        
        # Calculate current savings rate
        current_savings = income - total_expenses
        current_savings_rate = (current_savings / income) * 100
        
        # Categorize expenses
        needs = {}
        wants = {}
        savings = {}
        
        # This is a simplified categorization - in production we'd have a more sophisticated approach
        for category, amount in expenses.items():
            if category.lower() in ["rent", "mortgage", "utilities", "groceries", "insurance", "healthcare"]:
                needs[category] = amount
            elif category.lower() in ["investments", "retirement", "emergency fund"]:
                savings[category] = amount
            else:
                wants[category] = amount
        
        # Calculate totals for each category
        total_needs = sum(needs.values())
        total_wants = sum(wants.values())
        total_savings = sum(savings.values())
        
        # Calculate percentages
        needs_percentage = (total_needs / income) * 100
        wants_percentage = (total_wants / income) * 100
        savings_percentage = (total_savings / income) * 100
        
        # Apply the 50/30/20 rule as a general guideline
        # 50% for needs, 30% for wants, 20% for savings
        ideal_needs = income * 0.5
        ideal_wants = income * 0.3
        ideal_savings = income * 0.2
        
        # Generate recommendations
        recommendations = []
        
        if needs_percentage > 50:
            recommendations.append("Your essential expenses are higher than the recommended 50% of income. Consider reviewing your housing costs or other essential expenses to see if there are opportunities to reduce them.")
        
        if wants_percentage > 30:
            recommendations.append("Your discretionary spending is higher than the recommended 30% of income. Consider reducing spending in categories like entertainment, dining out, or subscriptions.")
        
        if savings_percentage < 20:
            recommendations.append("Your savings rate is below the recommended 20%. Consider increasing your savings by reducing discretionary spending or finding ways to increase income.")
        
        # If a specific savings target was provided
        if savings_target and current_savings < savings_target:
            shortfall = savings_target - current_savings
            recommendations.append(f"You're currently saving ${current_savings:.2f} per month, which is ${shortfall:.2f} short of your target. Consider reducing expenses in discretionary categories to reach your savings goal.")
        
        return {
            "current_budget": {
                "income": income,
                "expenses": expenses,
                "total_expenses": total_expenses,
                "current_savings": current_savings,
                "current_savings_rate": current_savings_rate
            },
            "budget_analysis": {
                "needs": needs,
                "wants": wants,
                "savings": savings,
                "needs_percentage": needs_percentage,
                "wants_percentage": wants_percentage,
                "savings_percentage": savings_percentage
            },
            "recommendations": recommendations
        }
    
    def _recommend_investments(self, savings: float, risk_tolerance: str, time_horizon: str, goals: List[str] = None) -> Dict[str, Any]:
        """Recommend investment allocations based on risk tolerance and time horizon"""
        # Define allocation models based on risk tolerance
        allocation_models = {
            "conservative": {
                "stocks": 0.30,
                "bonds": 0.50,
                "cash": 0.15,
                "alternative": 0.05
            },
            "moderate": {
                "stocks": 0.60,
                "bonds": 0.30,
                "cash": 0.05,
                "alternative": 0.05
            },
            "aggressive": {
                "stocks": 0.80,
                "bonds": 0.15,
                "cash": 0.00,
                "alternative": 0.05
            }
        }
        
        # Adjust allocations based on time horizon
        time_adjustments = {
            "short": {"stocks": -0.10, "bonds": 0.00, "cash": 0.10, "alternative": 0.00},
            "medium": {"stocks": 0.00, "bonds": 0.00, "cash": 0.00, "alternative": 0.00},
            "long": {"stocks": 0.10, "bonds": -0.05, "cash": -0.05, "alternative": 0.00}
        }
        
        # Get base allocation from risk tolerance
        if risk_tolerance.lower() not in allocation_models:
            risk_tolerance = "moderate"  # Default to moderate if invalid
            
        allocation = allocation_models[risk_tolerance.lower()].copy()
        
        # Apply time horizon adjustments
        if time_horizon.lower() in time_adjustments:
            for asset_class, adjustment in time_adjustments[time_horizon.lower()].items():
                allocation[asset_class] += adjustment
        
        # Ensure allocations are positive and sum to 1
        for asset_class in allocation:
            allocation[asset_class] = max(0, allocation[asset_class])
        
        total = sum(allocation.values())
        for asset_class in allocation:
            allocation[asset_class] /= total
        
        # Calculate dollar amounts
        dollar_allocation = {asset_class: savings * percentage for asset_class, percentage in allocation.items()}
        
        # Generate specific investment recommendations
        investment_recommendations = []
        
        if allocation["stocks"] > 0:
            investment_recommendations.append(f"Stocks (${dollar_allocation['stocks']:.2f}): Consider low-cost index funds like S&P 500 ETFs or total market ETFs.")
            
        if allocation["bonds"] > 0:
            investment_recommendations.append(f"Bonds (${dollar_allocation['bonds']:.2f}): Consider a mix of government and high-quality corporate bond funds.")
            
        if allocation["cash"] > 0:
            investment_recommendations.append(f"Cash (${dollar_allocation['cash']:.2f}): Hold in high-yield savings accounts or money market funds.")
            
        if allocation["alternative"] > 0:
            investment_recommendations.append(f"Alternative (${dollar_allocation['alternative']:.2f}): Consider REITs, commodities, or other alternative investments for diversification.")
        
        return {
            "allocation_percentages": allocation,
            "dollar_allocation": dollar_allocation,
            "investment_recommendations": investment_recommendations
        }
    
    def _plan_debt_repayment(self, debt: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create a debt repayment plan prioritizing high-interest debt"""
        # Sort debts by interest rate (highest to lowest)
        sorted_debts = sorted(debt.items(), key=lambda x: x[1]["interest_rate"], reverse=True)
        
        # Calculate total debt and monthly payment
        total_debt = sum(info["balance"] for _, info in sorted_debts)
        total_monthly_payment = sum(info["minimum_payment"] for _, info in sorted_debts)
        
        # Create avalanche method plan (highest interest first)
        avalanche_plan = []
        for debt_name, info in sorted_debts:
            avalanche_plan.append({
                "debt": debt_name,
                "balance": info["balance"],
                "interest_rate": info["interest_rate"],
                "minimum_payment": info["minimum_payment"],
                "priority": len(avalanche_plan) + 1
            })
        
        # Create snowball method plan (smallest balance first)
        snowball_sorted = sorted(debt.items(), key=lambda x: x[1]["balance"])
        snowball_plan = []
        for debt_name, info in snowball_sorted:
            snowball_plan.append({
                "debt": debt_name,
                "balance": info["balance"],
                "interest_rate": info["interest_rate"],
                "minimum_payment": info["minimum_payment"],
                "priority": len(snowball_plan) + 1
            })
        
        # Generate recommendations
        recommendations = [
            "Make minimum payments on all debts to avoid late fees and credit score damage.",
            "Focus extra payments on the highest interest debt first (avalanche method) to minimize interest costs.",
            "Once the highest interest debt is paid off, redirect those payments to the next highest interest debt.",
            "If you need psychological wins to stay motivated, consider the snowball method (smallest balance first).",
            "Consider consolidating high-interest debts if you can qualify for lower rates.",
            "Look for opportunities to increase income or reduce expenses to accelerate debt repayment."
        ]
        
        return {
            "total_debt": total_debt,
            "total_monthly_payment": total_monthly_payment,
            "avalanche_plan": avalanche_plan,
            "snowball_plan": snowball_plan,
            "recommendations": recommendations
        }
    
    def _calculate_savings_plan(self, goal_amount: float, time_frame_months: int, current_savings: float, monthly_contribution: float, estimated_return: float = 0.06) -> Dict[str, Any]:
        """Calculate a savings plan to reach a specific financial goal"""
        # Convert annual return to monthly
        monthly_return = (1 + estimated_return) ** (1/12) - 1
        
        # Calculate how much will be saved with current contribution
        future_value = current_savings * (1 + monthly_return) ** time_frame_months
        
        for i in range(time_frame_months):
            future_value += monthly_contribution * (1 + monthly_return) ** i
        
        # Determine if goal will be met
        goal_met = future_value >= goal_amount
        
        # If goal won't be met, calculate required monthly contribution
        required_contribution = monthly_contribution
        
        if not goal_met:
            # Use financial formula for future value of annuity
            # FV = P * ((1 + r)^n - 1) / r
            # Solve for P (payment)
            
            # First, calculate how much needs to come from new contributions
            needed_from_contributions = goal_amount - (current_savings * (1 + monthly_return) ** time_frame_months)
            
            # Then calculate the required monthly contribution
            if monthly_return > 0:
                required_contribution = needed_from_contributions * monthly_return / ((1 + monthly_return) ** time_frame_months - 1)
            else:
                required_contribution = needed_from_contributions / time_frame_months
        
        # Generate recommendations
        recommendations = []
        
        if goal_met:
            recommendations.append(f"Based on your current saving rate of ${monthly_contribution:.2f} per month, you're on track to reach your goal of ${goal_amount:.2f} in {time_frame_months} months.")
            recommendations.append(f"In fact, you're projected to save ${future_value - goal_amount:.2f} more than your goal.")
        else:
            recommendations.append(f"To reach your goal of ${goal_amount:.2f} in {time_frame_months} months, you need to save ${required_contribution:.2f} per month.")
            recommendations.append(f"This is ${required_contribution - monthly_contribution:.2f} more than your current monthly contribution.")
            
            # Provide additional recommendations if the required increase is significant
            if required_contribution > monthly_contribution * 1.5:
                recommendations.append("Consider extending your time frame or adjusting your goal amount if the required increase in savings is too challenging.")
                recommendations.append("Look for ways to reduce expenses or increase income to boost your savings rate.")
                recommendations.append("Consider higher-return investment options if appropriate for your risk tolerance and time horizon.")
        
        return {
            "goal_amount": goal_amount,
            "time_frame_months": time_frame_months,
            "current_savings": current_savings,
            "current_monthly_contribution": monthly_contribution,
            "estimated_annual_return": estimated_return,
            "projected_future_value": future_value,
            "goal_met": goal_met,
            "required_monthly_contribution": required_contribution,
            "recommendations": recommendations
        }
    
    def _optimize_tax_strategy(self, income: float, tax_brackets: Dict[str, Any], retirement_contributions: float = 0, hsa_contributions: float = 0, itemized_deductions: float = 0) -> Dict[str, Any]:
        """Suggest tax optimization strategies based on income and deductions"""
        # This is a simplified tax calculation and should not be used for actual tax planning
        # In production, we would use a more sophisticated tax calculation system
        
        standard_deduction = 12950  # 2022 standard deduction for single filers
        
        # Determine whether to use standard or itemized deduction
        use_itemized = itemized_deductions > standard_deduction
        deduction = itemized_deductions if use_itemized else standard_deduction
        
        # Calculate taxable income
        taxable_income = max(0, income - retirement_contributions - hsa_contributions - deduction)
        
        # Calculate current tax liability (simplified)
        tax_liability = 0
        remaining_income = taxable_income
        
        for bracket in sorted(tax_brackets.items(), key=lambda x: float(x[0])):
            rate = float(bracket[0]) / 100
            limit = bracket[1]
            
            if limit is None:  # Top bracket
                tax_liability += remaining_income * rate
                break
            
            if remaining_income <= 0:
                break
                
            taxable_in_bracket = min(remaining_income, limit)
            tax_liability += taxable_in_bracket * rate
            remaining_income -= taxable_in_bracket
        
        # Generate optimization recommendations
        recommendations = []
        
        # Retirement contribution recommendations
        max_401k = 20500  # 2022 401(k) limit
        if retirement_contributions < max_401k:
            potential_savings = min(max_401k - retirement_contributions, taxable_income) * 0.22  # Assume 22% marginal rate for simplicity
            recommendations.append(f"Consider increasing 401(k) contributions by up to ${max_401k - retirement_contributions:.2f} to reduce taxable income. This could save approximately ${potential_savings:.2f} in taxes.")
        
        # HSA recommendations
        max_hsa = 3650  # 2022 HSA limit for individual
        if hsa_contributions < max_hsa:
            potential_savings = min(max_hsa - hsa_contributions, taxable_income) * 0.22  # Assume 22% marginal rate for simplicity
            recommendations.append(f"If eligible, consider contributing up to ${max_hsa - hsa_contributions:.2f} more to an HSA. This could save approximately ${potential_savings:.2f} in taxes.")
        
        # Itemized deduction recommendations
        if not use_itemized and itemized_deductions > 0:
            difference = standard_deduction - itemized_deductions
            recommendations.append(f"You're currently better off taking the standard deduction of ${standard_deduction:.2f}, which is ${difference:.2f} more than your itemized deductions.")
            recommendations.append("Consider 'bunching' deductions in alternate years to maximize itemized deductions in certain years.")
        
        # Income deferral or acceleration
        recommendations.append("Consider deferring income to next year or accelerating deductions into this year if you expect to be in a lower tax bracket next year.")
        
        # Tax-efficient investing
        recommendations.append("Place tax-inefficient investments (like bonds) in tax-advantaged accounts and tax-efficient investments (like index funds) in taxable accounts.")
        recommendations.append("Consider tax-loss harvesting to offset capital gains with capital losses.")
        
        return {
            "current_tax_situation": {
                "gross_income": income,
                "retirement_contributions": retirement_contributions,
                "hsa_contributions": hsa_contributions,
                "itemized_deductions": itemized_deductions,
                "standard_deduction": standard_deduction,
                "deduction_used": "Itemized" if use_itemized else "Standard",
                "taxable_income": taxable_income,
                "estimated_tax_liability": tax_liability
            },
            "recommendations": recommendations
        }
