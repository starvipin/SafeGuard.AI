# Pipeline order: prepare data, train, evaluate, then manually upload to HF. Importing this package runs no stages.
"""Run steps 01 → 02 → 03, then explicitly publish with step 04.

This package is independent of the Flask web application.
"""
