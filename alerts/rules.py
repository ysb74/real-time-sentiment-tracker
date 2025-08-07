"""
Defines alert rules and thresholds for triggering notifications.
"""

import yaml
import os

class AlertRule:
    def __init__(self, name, sentiment, threshold, window, platforms, channels, message):
        self.name = name
        self.sentiment = sentiment
        self.threshold = threshold  # e.g., 10 negative posts
        self.window = window        # e.g., 5, # minutes
        self.platforms = platforms  # list of platforms
        self.channels = channels    # e.g. ['email', 'slack']
        self.message = message      # custom message template

def load_rules(path="config/alert_rules.yaml"):
    with open(path) as f:
        data = yaml.safe_load(f)
    rules = []
    for rule in data.get('rules', []):
        rules.append(AlertRule(**rule))
    return rules
