
from database import DatabaseManager
import json

def inspect_database():
    """Inspect the database structure and tables"""
    try:
        db = DatabaseManager()
        print("✅ Connected to database")
        
        # List tables (this is a simplified approach)
        # Note: Supabase client doesn't directly expose table listings, so we'll check our main tables
        tables_to_check = ["expenses", "goals"]
        
        for table in tables_to_check:
            try:
                # Try to get a single row to verify table exists
                response = db.supabase.table(table).select("*").limit(1).execute()
                print(f"✅ Table '{table}' exists")
                
                # Print schema based on the first row if available
                if response.data:
                    print(f"  Fields in '{table}': {list(response.data[0].keys())}")
                else:
                    print(f"  Table '{table}' exists but has no data")
                    
            except Exception as e:
                print(f"❌ Error accessing table '{table}': {str(e)}")
        
        print("\nTesting specific database operations:")
        
        # Test expenses retrieval
        try:
            expenses = db.get_expenses()
            print(f"✅ get_expenses(): Found {len(expenses)} records")
        except Exception as e:
            print(f"❌ get_expenses() failed: {str(e)}")
            
        # Test goals retrieval
        try:
            goals = db.get_goals()
            print(f"✅ get_goals(): Found {len(goals)} records")
        except Exception as e:
            print(f"❌ get_goals() failed: {str(e)}")
            
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")

if __name__ == "__main__":
    inspect_database()