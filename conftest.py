# Root conftest.py - Ensures repository root is added to sys.path during pytest execution
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
