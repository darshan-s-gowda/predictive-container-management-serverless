"""
Pytest configuration for the Predictive Container Management project.

Adds src/ to sys.path so all test modules can import project code without
needing an installed package.
"""
import sys
from pathlib import Path

# Insert src/ at the front of the path so imports resolve correctly.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
