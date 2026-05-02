#!/bin/bash
# Script to create a development user for Onyx
# Usage: ./create_dev_user.sh

DEV_EMAIL="a@test.com"
DEV_PASSWORD="a"

# Try frontend proxy first, fallback to direct backend
API_URL="${API_SERVER_URL:-http://localhost:3000/api}"

echo "Creating development user: $DEV_EMAIL"
echo "Using API URL: $API_URL"

response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$DEV_EMAIL\", \"username\": \"$DEV_EMAIL\", \"password\": \"$DEV_PASSWORD\"}")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    echo "✅ Successfully created development user!"
    echo ""
    echo "You can now login with:"
    echo "  Email: $DEV_EMAIL"
    echo "  Password: $DEV_PASSWORD"
    exit 0
elif [ "$http_code" -eq 400 ]; then
    if echo "$body" | grep -q "REGISTER_USER_ALREADY_EXISTS"; then
        echo "ℹ️  User $DEV_EMAIL already exists."
        echo ""
        echo "You can login with:"
        echo "  Email: $DEV_EMAIL"
        echo "  Password: $DEV_PASSWORD"
        exit 0
    else
        echo "❌ Failed to create user. Error: $body"
        exit 1
    fi
elif [ "$http_code" -eq 000 ]; then
    echo "❌ Error: Could not connect to the API server."
    echo "   Make sure the backend is running."
    echo "   Try: docker-compose up or check your backend logs"
    exit 1
else
    echo "❌ Failed to create user. Status: $http_code"
    echo "   Response: $body"
    exit 1
fi


















