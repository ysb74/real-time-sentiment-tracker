"""
Watches Redis for rule matches, triggers alerts.
"""
import time
from alerts.rules import load_rules
from alerts.dispatcher import dispatch_alert
import redis
from datetime import datetime, timedelta

class AlertMonitor:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url)
        self.rules = load_rules()

    def check_rule(self, rule):
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=rule.window)
        for platform in rule.platforms:
            key = f"sentiment:platform:{platform}"
            # Get all negative posts in window
            post_ids = self.redis.zrangebyscore(f"sentiment:label:{rule.sentiment}",
                                                window_start.timestamp(), now.timestamp())
            count = len(post_ids)
            if count >= rule.threshold:
                msg = rule.message.format(count=count, window=rule.window)
                dispatch_alert(rule.channels, f"ALERT: {rule.name}", msg)

    def run(self):
        while True:
            for rule in self.rules:
                self.check_rule(rule)
            time.sleep(30)  # Polling interval

if __name__ == "__main__":
    monitor = AlertMonitor(redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    monitor.run()
