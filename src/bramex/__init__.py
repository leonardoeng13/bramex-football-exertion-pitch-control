"""Shared pipeline for team BraMex (Parts 1 to 4 of the team notebook)."""
from .config import CONFIG, data_dir, cache_dir, setup_environment
from .data import discover_matches, load_raw_match
from .tables import build_match_tables, load_tables, Tables
from .checks import run_checks
from .registry import register_features, log_result, integrate, results_log, feature_registry, FEATURES, RESULTS
from .stats import within_effect, bh_fdr

__version__ = "0.1.0"
