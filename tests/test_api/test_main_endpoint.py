import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    """Test main prediction endpoint"""
    response = client.post(
        "/predict",
        json={"query": "Click here to verify your account: http://fake-bank.com"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "type_of_scam" in data
    assert "risk_score" in data
    assert "label" in data
    assert "explanation" in data

def test_predict_empty_query():
    """Test prediction with empty query"""
    response = client.post(
        "/predict",
        json={"query": ""}
    )
    assert response.status_code == 200  # Should handle gracefully

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_home_page():
    """Test home page loads"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
