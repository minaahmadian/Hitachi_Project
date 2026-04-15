import os

def dumb_pre_filter(raw_text: str) -> str:
    """Basic Python filter (No AI). Searches for critical keywords to save tokens."""
    print("Dumb Filter: Scanning strings in progress (No LLM)...")
    
    danger_keywords = [
        "workaround", "bypass", "disabled", "temporary", 
        "manually", "not integrated", "new function", 
        "false positive", "skip tests", "ignore warning"
    ]
    
    emails = raw_text.split("--- EMAIL")
    suspicious_emails = []
    
    for email in emails:
        if not email.strip(): 
            continue
        
        email_lower = email.lower()
        if any(keyword in email_lower for keyword in danger_keywords):
            suspicious_emails.append(f"--- EMAIL {email}")
            
    filtered_text = "".join(suspicious_emails)
    
    if filtered_text:
        print(f"   -> Filtering completed: Found {len(suspicious_emails)} suspicious emails.")
        return filtered_text
    else:
        print("   -> No critical words found. Stopping email pipeline.")
        return "No anomalies detected in text."

def load_local_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    try:
        with open(os.path.join(data_dir, "test_logs.json"), "r", encoding="utf-8") as f:
            logs_data = f.read()
    except FileNotFoundError:
        logs_data = "{}"
        
    try:
        with open(os.path.join(data_dir, "email_threads.txt"), "r", encoding="utf-8") as f:
            raw_email_data = f.read()
            email_data = dumb_pre_filter(raw_email_data)
    except FileNotFoundError:
        email_data = "No emails found."
        
    return logs_data, email_data