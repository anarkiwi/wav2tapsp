import os
import sys

# Make the src-layout package importable when running the tests without an
# editable install (CI installs `-e .[test]`, but this keeps `pytest` working
# straight from a checkout too).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
