#!/usr/bin/env python3
"""Module for configuring and providing the application logger."""
import dotenv
import logging
import os

dotenv.load_dotenv()

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
