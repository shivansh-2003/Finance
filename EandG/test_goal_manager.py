# test_goal_manager.py
import unittest
from unittest.mock import patch, MagicMock
from goal_manager import GoalManager

class TestGoalManager(unittest.TestCase):
    
    @patch('goal_manager.DatabaseManager')
    @patch('goal_manager.NLPProcessor')
    def test_process_goal_input(self, MockNLPProcessor, MockDatabaseManager):
        # Arrange
        manager = GoalManager()
        mock_nlp = MockNLPProcessor.return_value
        mock_db = MockDatabaseManager.return_value
        
        # Simulate NLP processing
        mock_nlp.extract_goal_details.return_value = {
            "target_amount": 1000,
            "purpose": "save for vacation",
            "deadline": "2023-12-31"  # Example deadline
        }
        
        # Simulate database add goal success
        mock_db.add_goal.return_value = [{"id": 1}]
        
        # Act
        result = manager.process_goal_input("I want to save $1000 for vacation by 2023-12-31.")
        print(result) 
        # Assert
        self.assertTrue(result["success"])
        self.assertIn("Goal added: Save $1000 for save for vacation by 2023-12-31", result["message"])

if __name__ == '__main__':
    unittest.main()