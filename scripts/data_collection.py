#!/usr/bin/env python3
"""
Data collection script for training ML models
"""

import pandas as pd
import requests
from pathlib import Path
import json

def create_sample_datasets():
    """Create sample datasets for training ML models"""
    
    # URL Scam Dataset
    url_scam_data = [
        {"text": "Your bank account has been suspended. Click here to verify: http://fake-bank.com", "label": 1},
        {"text": "Urgent: Verify your PayPal account: http://paypal-security.fake.com", "label": 1},
        {"text": "Netflix subscription expired. Renew here: http://netflix-renewal.scam.com", "label": 1},
        {"text": "Check out this interesting article: https://www.bbc.com/news/technology", "label": 0},
        {"text": "Here's the document you requested: https://drive.google.com/file/d/abc123", "label": 0},
        {"text": "Meeting link: https://zoom.us/j/1234567890", "label": 0},
    ]
    
    # QR Code Scam Dataset
    qr_scam_data = [
        {"text": "Congratulations! You won $500! Scan this QR code to claim your prize now!", "label": 1},
        {"text": "Free WiFi! Scan QR code for instant access. No password needed!", "label": 1},
        {"text": "Scan this QR code to download our exclusive app and get free rewards!", "label": 1},
        {"text": "Restaurant menu QR code. Scan to view our dishes and prices.", "label": 0},
        {"text": "Scan QR code to connect to our guest WiFi network.", "label": 0},
        {"text": "Event ticket QR code. Please scan at entrance for verification.", "label": 0},
    ]
    
    # Payment Request Scam Dataset
    payment_data = [
        {"text": "Hi, this is your cousin. I'm in emergency and need $200 urgently. Send to 9876543210", "label": 1},
        {"text": "Your friend John is stuck at airport. Send ₹5000 immediately to help him.", "label": 1},
        {"text": "Urgent medical emergency! Need money transfer right now. Family member in hospital.", "label": 1},
        {"text": "Can you transfer the money for groceries? I'll pay you back when I see you.", "label": 0},
        {"text": "Monthly rent payment due. Please transfer to landlord account as discussed.", "label": 0},
        {"text": "Splitting dinner bill. Send me ₹500 for your share. Thanks!", "label": 0},
    ]
    
    # Investment Scam Dataset
    investment_data = [
        {"text": "Guaranteed 300% returns in 30 days! Join our crypto trading group. Risk-free profits!", "label": 1},
        {"text": "Make ₹50,000 daily from home! Our forex bot has 100% success rate. No experience needed!", "label": 1},
        {"text": "Double your money in 24 hours! Limited time investment opportunity. Act now!", "label": 1},
        {"text": "Consider diversifying your portfolio with index funds for long-term growth.", "label": 0},
        {"text": "Our mutual fund has delivered consistent 12% annual returns over 10 years.", "label": 0},
        {"text": "Investment advisory: Market volatility suggests cautious approach this quarter.", "label": 0},
    ]
    
    # Create DataFrames and save
    datasets = {
        "url_scam_data.csv": pd.DataFrame(url_scam_data),
        "qr_scam_data.csv": pd.DataFrame(qr_scam_data),
        "payment_data.csv": pd.DataFrame(payment_data),
        "investment_data.csv": pd.DataFrame(investment_data)
    }
    
    # Ensure directory exists
    data_dir = Path("ml_models/data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    for filename, df in datasets.items():
        filepath = data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"✅ Created {filepath}")
    
    print(f"\n📊 Sample datasets created with {sum(len(df) for df in datasets.values())} total samples")
    print("💡 These are minimal datasets for demonstration. For production, collect more diverse data.")

def download_public_datasets():
    """Download public scam datasets if available"""
    print("🌐 Checking for public scam datasets...")
    
    # This is a placeholder for actual dataset URLs
    # In practice, you would download from legitimate sources
    public_sources = [
        # "https://example.com/phishing-dataset.csv",
        # "https://example.com/scam-messages.json"
    ]
    
    if not public_sources:
        print("ℹ️  No public datasets configured. Using sample data only.")
        return
    
    for url in public_sources:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                filename = Path(url).name
                filepath = Path("ml_models/data/datasets") / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Downloaded {filename}")
            else:
                print(f"❌ Failed to download {url}")
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")

if __name__ == "__main__":
    print("📊 Starting data collection for scam detection models...")
    
    create_sample_datasets()
    download_public_datasets()
    
    print("\n🎉 Data collection complete!")
    print("Next step: Run training scripts in ml_models/training/")
