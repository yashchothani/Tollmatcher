"""
TollMatcher Speech Translation System
Package root initialization.
"""

import os
import sys

# Ensure root directory is always on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
