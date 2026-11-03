"""Shared builders for tests: native stubs, imagination fixtures and fakes.

Test modules import from here, never from each other.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
