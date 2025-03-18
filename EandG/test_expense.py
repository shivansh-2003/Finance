# test_expense.py
import unittest
from unittest.mock import patch, MagicMock
from expense_tracker import ExpenseTracker

class TestExpenseTracker(unittest.TestCase):
    
    @patch('expense_tracker.DatabaseManager')
    @patch('expense_tracker.NLPProcessor')
    def test_process_expense_input(self, MockNLPProcessor, MockDatabaseManager):
        # Arrange
        tracker = ExpenseTracker()
        mock_nlp = MockNLPProcessor.return_value
        mock_db = MockDatabaseManager.return_value
        
        # Simulate NLP processing
        mock_nlp.extract_expense_details.return_value = {
            "amount": 45,
            "category": "groceries",
            "date": "2023-10-01",  # Example date
            "description": "groceries"
        }
        
        # Simulate database add expense success
        mock_db.add_expense.return_value = True
        
        # Act
        result = tracker.process_expense_input("I spent $45 on groceries yesterday.")
        print(result) 
        # Assert
        self.assertTrue(result["success"])
        self.assertIn("Expense added: $45 for groceries in category groceries", result["message"])

if __name__ == '__main__':
    unittest.main()