import os
import sys

# Make the repo-root modules (wav2tapsp, c64tape) importable from tests/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
