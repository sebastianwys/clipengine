# run_tests.py
# run the whole suite with the project venv, from the project root:
#   .venv/bin/python run_tests.py

import sys
import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
