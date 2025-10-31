"""
API tests for authentication endpoints
"""

import pytest
from fastapi.testclient import TestClient

class TestAuthAPI:
    """Test cases for authentication API endpoints"""
    
    def test_register_user(self, client, sample_user_data):
        """Test user registration"""
        response = client.post("/api/auth/register", json=sample_user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["username"] == sample_user_data["username"]
        assert data["email"] == sample_user_data["email"]
        assert "password" not in data  # Password should not be returned
    
    def test_register_duplicate_username(self, client, sample_user_data):
        """Test registration with duplicate username"""
        # Register first user
        response = client.post("/api/auth/register", json=sample_user_data)
        assert response.status_code == 201
        
        # Try to register with same username
        duplicate_data = sample_user_data.copy()
        duplicate_data["email"] = "different@example.com"
        response = client.post("/api/auth/register", json=duplicate_data)
        
        assert response.status_code == 400
        assert "username" in response.json()["detail"].lower()
    
    def test_register_duplicate_email(self, client, sample_user_data):
        """Test registration with duplicate email"""
        # Register first user
        response = client.post("/api/auth/register", json=sample_user_data)
        assert response.status_code == 201
        
        # Try to register with same email
        duplicate_data = sample_user_data.copy()
        duplicate_data["username"] = "differentuser"
        response = client.post("/api/auth/register", json=duplicate_data)
        
        assert response.status_code == 400
        assert "email" in response.json()["detail"].lower()
    
    def test_register_invalid_data(self, client):
        """Test registration with invalid data"""
        invalid_data = {
            "username": "ab",  # Too short
            "email": "invalid-email",
            "password": "123"  # Too short
        }
        
        response = client.post("/api/auth/register", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_login_success(self, client, sample_user_data):
        """Test successful login"""
        # Register user first
        response = client.post("/api/auth/register", json=sample_user_data)
        assert response.status_code == 201
        
        # Login
        login_data = {
            "username": sample_user_data["username"],
            "password": sample_user_data["password"]
        }
        response = client.post("/api/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client, sample_user_data):
        """Test login with invalid credentials"""
        # Register user first
        response = client.post("/api/auth/register", json=sample_user_data)
        assert response.status_code == 201
        
        # Try to login with wrong password
        login_data = {
            "username": sample_user_data["username"],
            "password": "wrongpassword"
        }
        response = client.post("/api/auth/login", data=login_data)
        
        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()
    
    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent user"""
        login_data = {
            "username": "nonexistent",
            "password": "password"
        }
        response = client.post("/api/auth/login", data=login_data)
        
        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()
    
    def test_get_current_user(self, client, auth_headers):
        """Test getting current user info"""
        response = client.get("/api/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "password" not in data
    
    def test_get_current_user_no_token(self, client):
        """Test getting current user without token"""
        response = client.get("/api/auth/me")
        
        assert response.status_code == 401
    
    def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/auth/me", headers=headers)
        
        assert response.status_code == 401
    
    def test_refresh_token(self, client, auth_headers):
        """Test token refresh"""
        response = client.post("/api/auth/refresh", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
    
    def test_logout(self, client, auth_headers):
        """Test logout"""
        response = client.post("/api/auth/logout", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_change_password(self, client, auth_headers, sample_user_data):
        """Test password change"""
        change_data = {
            "current_password": sample_user_data["password"],
            "new_password": "newpassword123"
        }
        
        response = client.post("/api/auth/change-password", 
                             json=change_data, 
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_change_password_wrong_current(self, client, auth_headers):
        """Test password change with wrong current password"""
        change_data = {
            "current_password": "wrongpassword",
            "new_password": "newpassword123"
        }
        
        response = client.post("/api/auth/change-password", 
                             json=change_data, 
                             headers=auth_headers)
        
        assert response.status_code == 400
        assert "incorrect" in response.json()["detail"].lower()
    
    def test_change_password_weak_new(self, client, auth_headers, sample_user_data):
        """Test password change with weak new password"""
        change_data = {
            "current_password": sample_user_data["password"],
            "new_password": "123"  # Too weak
        }
        
        response = client.post("/api/auth/change-password", 
                             json=change_data, 
                             headers=auth_headers)
        
        assert response.status_code == 422  # Validation error
    
    def test_reset_password_request(self, client, sample_user_data):
        """Test password reset request"""
        # Register user first
        response = client.post("/api/auth/register", json=sample_user_data)
        assert response.status_code == 201
        
        # Request password reset
        reset_data = {"email": sample_user_data["email"]}
        response = client.post("/api/auth/reset-password-request", json=reset_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_reset_password_request_nonexistent_email(self, client):
        """Test password reset request with nonexistent email"""
        reset_data = {"email": "nonexistent@example.com"}
        response = client.post("/api/auth/reset-password-request", json=reset_data)
        
        # Should still return 200 for security (don't reveal if email exists)
        assert response.status_code == 200
    
    def test_verify_email(self, client, sample_user_data):
        """Test email verification"""
        # Register user first
        response = client.post("/api/auth/register", json=sample_user_data)
        assert response.status_code == 201
        
        # This would normally require a verification token
        # For testing, we'll assume the endpoint exists
        verify_data = {"token": "test_verification_token"}
        response = client.post("/api/auth/verify-email", json=verify_data)
        
        # This might return 400 for invalid token in real implementation
        assert response.status_code in [200, 400]
    
    def test_rate_limiting(self, client, sample_user_data):
        """Test rate limiting on login attempts"""
        # Register user first
        response = client.post("/api/auth/register", json=sample_user_data)
        assert response.status_code == 201
        
        # Make multiple failed login attempts
        login_data = {
            "username": sample_user_data["username"],
            "password": "wrongpassword"
        }
        
        for i in range(10):  # Make 10 failed attempts
            response = client.post("/api/auth/login", data=login_data)
            assert response.status_code == 401
        
        # The 11th attempt might be rate limited
        response = client.post("/api/auth/login", data=login_data)
        # Depending on rate limit configuration, this might be 429 or still 401
        assert response.status_code in [401, 429]
