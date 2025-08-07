"""
Dispatches alert notifications to configured channels.
"""

import smtplib
import os
from email.mime.text import MIMEText
# For Slack or others: use requests, httpx, or official SDKs

def send_email(subject, body, to_addr):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = os.environ["ALERT_EMAIL_FROM"]
    msg['To'] = to_addr

    with smtplib.SMTP(os.environ["SMTP_SERVER"], 587) as server:
        server.starttls()
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        server.send_message(msg)

def send_slack(message, webhook_url):
    import requests
    requests.post(webhook_url, json={"text": message})

def dispatch_alert(channels, subject, message):
    for channel in channels:
        if channel == "email":
            send_email(subject, message, os.environ["ALERT_EMAIL_TO"])
        if channel == "slack":
            send_slack(message, os.environ["SLACK_WEBHOOK_URL"])
