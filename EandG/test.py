# supabase_test_integer_categories.py
from database import DatabaseManager
from datetime import datetime, timedelta
import time
import sys

def test_database_connection():
    """Test that the database connection works"""
    try:
        db = DatabaseManager()
        print("✅ Database connection successful")
        return db
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return None

def add_test_data(db):
    """Add sample test data to the database"""
    print("\nAdding test data to Supabase...")
    
    # Generate unique test data with timestamps to easily identify test runs
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Sample expenses with integer categories
    # Category IDs: 1=groceries, 2=dining, 3=utilities
    expenses = [
        {"amount": 45.50, "category": 1, "date": datetime.now().strftime("%Y-%m-%d"), 
         "description": f"Test grocery shopping {timestamp}"},
        {"amount": 30.00, "category": 2, "date": datetime.now().strftime("%Y-%m-%d"), 
         "description": f"Test dinner {timestamp}"},
        {"amount": 120.00, "category": 3, "date": datetime.now().strftime("%Y-%m-%d"), 
         "description": f"Test utility bill {timestamp}"}
    ]

    # Add expenses to the database
    expense_ids = []
    for expense in expenses:
        try:
            result = db.add_expense(expense["amount"], expense["category"], expense["date"], expense["description"])
            if result:
                expense_ids.append(result[0]['id'])
                print(f"✅ Added expense: ${expense['amount']} for {expense['description']}")
            else:
                print(f"❌ Failed to add expense: {expense}")
        except Exception as e:
            print(f"❌ Error adding expense: {str(e)}")
            print(f"   Expense data: {expense}")

    # Sample goal with timestamp
    goal_deadline = (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d")
    goal_purpose = f"Test Vacation {timestamp}"
    
    try:
        result = db.add_goal(1000.00, goal_purpose, goal_deadline)
        if result:
            goal_id = result[0]['id']
            print(f"✅ Added goal: $1000 for {goal_purpose} by {goal_deadline}")
        else:
            print("❌ Failed to add goal")
    except Exception as e:
        print(f"❌ Error adding goal: {str(e)}")
    
    print(f"\nAdded {len(expense_ids)} expenses with timestamp {timestamp}")
    return expense_ids, timestamp

def verify_data_exists(db, timestamp):
    """Verify that the test data exists in the database"""
    print("\nVerifying data in Supabase...")
    
    try:
        # Wait a moment for Supabase to process the data
        time.sleep(1)
        
        # Get all expenses
        expenses = db.get_expenses()
        
        # Check if expenses exist
        if expenses:
            # Filter expenses by the timestamp in the description
            test_expenses = [e for e in expenses if timestamp in e.get("description", "")]
            print(f"✅ Found {len(test_expenses)} test expenses in the database")
            
            # Print details of found expenses
            for i, expense in enumerate(test_expenses[:3]):  # Limit to first 3 for brevity
                print(f"  Expense {i+1}: ${expense['amount']} - {expense['description']}")
            
            if len(test_expenses) == 0:
                print("❌ Could not find any test expenses with the current timestamp")
        else:
            print("❌ No expenses found in the database")
            test_expenses = []
            
        # Get all goals
        goals = db.get_goals()
        
        # Check if goals exist
        if goals:
            # Filter goals by the timestamp in the purpose
            test_goals = [g for g in goals if timestamp in g.get("purpose", "")]
            print(f"✅ Found {len(test_goals)} test goals in the database")
            
            # Print details of found goals
            for i, goal in enumerate(test_goals[:3]):  # Limit to first 3 for brevity
                print(f"  Goal {i+1}: ${goal['target_amount']} for {goal['purpose']}")
                
            if len(test_goals) == 0:
                print("❌ Could not find any test goals with the current timestamp")
        else:
            print("❌ No goals found in the database")
            test_goals = []
            
        return len(test_expenses) > 0 and len(test_goals) > 0
        
    except Exception as e:
        print(f"❌ Error verifying data: {str(e)}")
        return False

def run_test():
    """Run the complete test"""
    print("=" * 50)
    print("SUPABASE DATABASE TEST")
    print("=" * 50)
    
    # Test database connection
    db = test_database_connection()
    if not db:
        print("❌ Test failed: Could not connect to database")
        return False
    
    # Add test data
    expense_ids, timestamp = add_test_data(db)
    
    # Verify data exists
    result = verify_data_exists(db, timestamp)
    
    # Print final result
    print("\n" + "=" * 50)
    if result:
        print("✅ TEST PASSED: Successfully added and verified data in Supabase")
    else:
        print("❌ TEST FAILED: Could not verify data in Supabase")
    print("=" * 50)
    
    return result

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)