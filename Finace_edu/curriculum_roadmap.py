"""
COMPREHENSIVE PERSONAL FINANCE CURRICULUM ROADMAP

This file defines the complete structured roadmap for teaching personal finance
from basics to advanced concepts. It's organized in modules of increasing difficulty.
"""

# The curriculum is structured in 5 levels of difficulty
# Level 1: Absolute Beginner
# Level 2: Basic Understanding
# Level 3: Intermediate Knowledge
# Level 4: Advanced Concepts
# Level 5: Expert-Level Strategies

CURRICULUM_ROADMAP = {
    "level_1": {
        "title": "Personal Finance Foundations",
        "description": "Learn the absolute basics of managing money and understanding financial terms.",
        "modules": [
            {
                "module_id": "module_1_1",
                "title": "Money Basics",
                "description": "Understanding the fundamental concepts of money and personal finance",
                "topics": [
                    "What is money and its functions",
                    "Income vs. expenses",
                    "Financial goals and values",
                    "Basic financial terminology"
                ],
                "exercises": [
                    "Track all your expenses for one week",
                    "Identify your income sources",
                    "Set three short-term financial goals"
                ],
                "estimated_time": "2-3 hours"
            },
            {
                "module_id": "module_1_2",
                "title": "Budgeting 101",
                "description": "Creating and maintaining your first personal budget",
                "topics": [
                    "What is a budget and why you need one",
                    "Fixed vs. variable expenses",
                    "Income categories",
                    "Basic budgeting methods",
                    "Common budgeting mistakes"
                ],
                "exercises": [
                    "Create your first monthly budget",
                    "Categorize your expenses from last month",
                    "Find three areas to reduce spending"
                ],
                "estimated_time": "3-4 hours"
            },
            {
                "module_id": "module_1_3",
                "title": "Banking Essentials",
                "description": "Understanding basic banking services and products",
                "topics": [
                    "Types of bank accounts",
                    "Checking vs. savings accounts",
                    "Banking fees and how to avoid them",
                    "Online and mobile banking basics",
                    "Debit cards and ATM usage"
                ],
                "exercises": [
                    "Review your bank statement and identify all fees",
                    "Set up online banking if you haven't already",
                    "Compare features of two different banks"
                ],
                "estimated_time": "2-3 hours"
            }
        ],
        "assessment": {
            "quiz_id": "quiz_level_1",
            "title": "Level 1 Assessment",
            "description": "Test your understanding of personal finance basics",
            "passing_score": 70
        }
    },
    "level_2": {
        "title": "Building Financial Security",
        "description": "Establish good financial habits and create your first safety nets.",
        "modules": [
            {
                "module_id": "module_2_1",
                "title": "Emergency Fund Building",
                "description": "Creating your financial safety net",
                "topics": [
                    "What is an emergency fund",
                    "How much to save for emergencies",
                    "Where to keep your emergency fund",
                    "Strategies to build your fund quickly",
                    "When to use your emergency fund"
                ],
                "exercises": [
                    "Calculate your ideal emergency fund size",
                    "Set up a separate savings account for emergencies",
                    "Create a plan to fully fund your emergency fund"
                ],
                "estimated_time": "2-3 hours"
            },
            {
                "module_id": "module_2_2",
                "title": "Debt Management",
                "description": "Understanding and managing different types of debt",
                "topics": [
                    "Good debt vs. bad debt",
                    "Interest rates and how they work",
                    "Credit cards and how to use them responsibly",
                    "Student loans overview",
                    "Basic debt repayment strategies"
                ],
                "exercises": [
                    "List all your debts with interest rates and minimum payments",
                    "Calculate how much interest you're paying monthly",
                    "Create a basic debt payoff plan"
                ],
                "estimated_time": "3-4 hours"
            },
            {
                "module_id": "module_2_3",
                "title": "Credit Scores",
                "description": "Understanding credit reports and building good credit",
                "topics": [
                    "What is a credit score",
                    "Factors that affect your credit score",
                    "How to check your credit report",
                    "Building credit from scratch",
                    "Improving your credit score"
                ],
                "exercises": [
                    "Get your free credit report",
                    "Identify factors hurting your credit score",
                    "Create a plan to improve your score by 50 points"
                ],
                "estimated_time": "2-3 hours"
            }
        ],
        "assessment": {
            "quiz_id": "quiz_level_2",
            "title": "Level 2 Assessment",
            "description": "Test your understanding of building financial security",
            "passing_score": 75
        }
    },
    "level_3": {
        "title": "Growing Your Wealth",
        "description": "Expand your financial knowledge and begin building wealth through investing.",
        "modules": [
            {
                "module_id": "module_3_1",
                "title": "Introduction to Investing",
                "description": "Understanding the basics of investing and compound growth",
                "topics": [
                    "What is investing and why it's important",
                    "Risk vs. return",
                    "Time value of money",
                    "Compound interest",
                    "Types of investment accounts",
                    "Basic asset classes (stocks, bonds, cash)"
                ],
                "exercises": [
                    "Calculate compound interest with different rates",
                    "Determine your risk tolerance",
                    "Research different investment account types"
                ],
                "estimated_time": "4-5 hours"
            },
            {
                "module_id": "module_3_2",
                "title": "Retirement Planning Basics",
                "description": "Starting to plan for your future retirement",
                "topics": [
                    "Why save for retirement early",
                    "Types of retirement accounts (401(k), IRA)",
                    "Employer matches",
                    "Traditional vs. Roth accounts",
                    "Basic retirement calculations"
                ],
                "exercises": [
                    "Calculate how much you need to save for retirement",
                    "Check if your employer offers retirement benefits",
                    "Set up or optimize a retirement account"
                ],
                "estimated_time": "3-4 hours"
            },
            {
                "module_id": "module_3_3",
                "title": "Tax Basics",
                "description": "Understanding how taxes work and basic tax strategies",
                "topics": [
                    "Income tax basics",
                    "Tax brackets and marginal tax rates",
                    "Deductions vs. credits",
                    "Tax-advantaged accounts",
                    "Basic tax filing information"
                ],
                "exercises": [
                    "Estimate your effective tax rate",
                    "Identify potential tax deductions for your situation",
                    "Create a simple tax planning strategy"
                ],
                "estimated_time": "3-4 hours"
            }
        ],
        "assessment": {
            "quiz_id": "quiz_level_3",
            "title": "Level 3 Assessment",
            "description": "Test your understanding of investing and wealth growth",
            "passing_score": 75
        }
    },
    "level_4": {
        "title": "Advanced Financial Strategies",
        "description": "Optimize your finances and implement sophisticated strategies.",
        "modules": [
            {
                "module_id": "module_4_1",
                "title": "Advanced Investment Strategies",
                "description": "Taking your investment knowledge to the next level",
                "topics": [
                    "Asset allocation and diversification",
                    "Index vs. active investing",
                    "Dollar-cost averaging",
                    "Rebalancing strategies",
                    "Tax-efficient investing"
                ],
                "exercises": [
                    "Create a diversified investment plan",
                    "Analyze fees in your current investments",
                    "Develop a rebalancing strategy"
                ],
                "estimated_time": "4-5 hours"
            },
            {
                "module_id": "module_4_2",
                "title": "Real Estate Investing",
                "description": "Understanding real estate as an investment vehicle",
                "topics": [
                    "Types of real estate investments",
                    "REITs vs. direct ownership",
                    "Rental property analysis",
                    "Financing investment properties",
                    "Tax implications of real estate"
                ],
                "exercises": [
                    "Analyze a potential rental property",
                    "Compare REITs to other investments",
                    "Calculate potential returns on a property"
                ],
                "estimated_time": "4-5 hours"
            },
            {
                "module_id": "module_4_3",
                "title": "Insurance Planning",
                "description": "Creating a comprehensive insurance strategy",
                "topics": [
                    "Types of insurance (life, health, disability, property)",
                    "Insurance needs analysis",
                    "Self-insurance vs. commercial policies",
                    "Policy evaluation and selection",
                    "Insurance as financial protection"
                ],
                "exercises": [
                    "Conduct a personal insurance audit",
                    "Calculate your life insurance needs",
                    "Compare insurance policies and features"
                ],
                "estimated_time": "3-4 hours"
            }
        ],
        "assessment": {
            "quiz_id": "quiz_level_4",
            "title": "Level 4 Assessment",
            "description": "Test your understanding of advanced financial strategies",
            "passing_score": 80
        }
    },
    "level_5": {
        "title": "Financial Mastery",
        "description": "Master complex financial concepts and optimize your financial life.",
        "modules": [
            {
                "module_id": "module_5_1",
                "title": "Estate Planning",
                "description": "Preparing your finances for the future and legacy planning",
                "topics": [
                    "Wills and trusts",
                    "Power of attorney",
                    "Advanced healthcare directives",
                    "Estate taxes",
                    "Inheritance planning",
                    "Charitable giving strategies"
                ],
                "exercises": [
                    "Create a basic estate plan outline",
                    "Research estate laws in your state",
                    "Identify potential estate planning professionals"
                ],
                "estimated_time": "4-5 hours"
            },
            {
                "module_id": "module_5_2",
                "title": "Advanced Tax Strategies",
                "description": "Sophisticated approaches to tax optimization",
                "topics": [
                    "Tax-loss harvesting",
                    "Charitable tax strategies",
                    "Business tax structures",
                    "International tax considerations",
                    "Estate tax planning",
                    "Tax planning across multiple years"
                ],
                "exercises": [
                    "Identify tax-loss harvesting opportunities",
                    "Plan a multi-year tax strategy",
                    "Analyze potential tax-advantaged investments"
                ],
                "estimated_time": "4-5 hours"
            },
            {
                "module_id": "module_5_3",
                "title": "Financial Independence Planning",
                "description": "Strategies for achieving financial freedom",
                "topics": [
                    "FIRE movement (Financial Independence, Retire Early)",
                    "Creating passive income streams",
                    "Safe withdrawal rates",
                    "Health care in early retirement",
                    "Psychological aspects of financial independence",
                    "Portfolio strategies for financial independence"
                ],
                "exercises": [
                    "Calculate your financial independence number",
                    "Identify potential passive income sources",
                    "Create a financial independence timeline"
                ],
                "estimated_time": "4-5 hours"
            }
        ],
        "assessment": {
            "quiz_id": "quiz_level_5",
            "title": "Level 5 Assessment",
            "description": "Test your mastery of advanced financial concepts",
            "passing_score": 85
        }
    }
}

# Function to get all modules in the curriculum
def get_all_modules():
    modules = []
    for level, level_data in CURRICULUM_ROADMAP.items():
        for module in level_data["modules"]:
            module_with_level = module.copy()
            module_with_level["level"] = level.split("_")[1]
            module_with_level["level_title"] = level_data["title"]
            modules.append(module_with_level)
    return modules

# Function to get modules by level
def get_modules_by_level(level):
    level_key = f"level_{level}"
    if level_key in CURRICULUM_ROADMAP:
        return CURRICULUM_ROADMAP[level_key]["modules"]
    return []

# Function to get a specific module by ID
def get_module_by_id(module_id):
    for level, level_data in CURRICULUM_ROADMAP.items():
        for module in level_data["modules"]:
            if module["module_id"] == module_id:
                return module
    return None

# Function to get the next recommended module after completing the current one
def get_next_module(current_module_id):
    all_modules = get_all_modules()
    for i, module in enumerate(all_modules):
        if module["module_id"] == current_module_id and i < len(all_modules) - 1:
            return all_modules[i + 1]
    return None

# Function to get assessment for a level
def get_level_assessment(level):
    level_key = f"level_{level}"
    if level_key in CURRICULUM_ROADMAP:
        return CURRICULUM_ROADMAP[level_key]["assessment"]
    return None 