import unittest
import calc

class TestCalc(unittest.TestCase):
    # Test inputs which should be valid
    def test_triangular_number(self):
        self.assertEqual(calc.triangular_numbers(1), 1)
        self.assertEqual(calc.triangular_numbers(2), 3)
        self.assertEqual(calc.triangular_numbers(99),4950)
        self.assertEqual(calc.triangular_numbers(100),5050)

    def test_invalid_input(self):
        # Test invalid inputs which should raise an error
        with self.assertRaises(ValueError):
            calc.triangular_numbers(0)
        with self.assertRaises(ValueError):
            calc.triangular_numbers(-10)
        with self.assertRaises(ValueError):
            calc.triangular_numbers(99.7)

if __name__ == "__main__":
    unittest.main()