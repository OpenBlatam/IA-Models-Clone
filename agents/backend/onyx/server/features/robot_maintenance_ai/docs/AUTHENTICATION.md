# Authentication Guide

## Overview

The Robot Maintenance AI system includes a basic authentication system using API keys. This allows you to control access to the API and track usage by user.

## API Keys

### Creating an API Key

```bash
curl -X POST http://localhost:8000/api/auth/api-key/create \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "permissions": ["read", "write"]
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "api_key": "rmai_abc123...",
    "user_id": "user123",
    "permissions": ["read", "write"]
  }
}
```

### Using an API Key

Include the API key in the `Authorization` header:

```bash
curl -X GET http://localhost:8000/api/notifications/user123 \
  -H "Authorization: Bearer rmai_abc123..."
```

### Validating an API Key

```bash
curl -X POST http://localhost:8000/api/auth/api-key/validate \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "rmai_abc123..."
  }'
```

### Revoking an API Key

```bash
curl -X POST http://localhost:8000/api/auth/api-key/revoke \
  -H "Authorization: Bearer rmai_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "rmai_abc123..."
  }'
```

## Permissions

Available permissions:
- `read`: Read access to resources
- `write`: Write access to resources
- `admin`: Administrative access

## Security Best Practices

1. **Store API keys securely**: Never commit API keys to version control
2. **Rotate keys regularly**: Revoke and create new keys periodically
3. **Use environment variables**: Store keys in environment variables
4. **Limit permissions**: Only grant necessary permissions
5. **Monitor usage**: Track API key usage for security

## Example: Protected Endpoint

Endpoints that require authentication use the `require_auth` dependency:

```python
@router.get("/protected")
async def protected_endpoint(
    user_info: Dict = Depends(require_auth)
):
    return {
        "message": f"Hello {user_info['user_id']}",
        "permissions": user_info["permissions"]
    }
```

## Production Considerations

For production, consider:
- Using a proper authentication service (OAuth2, JWT)
- Implementing token expiration
- Adding rate limiting per API key
- Logging all authentication attempts
- Using HTTPS only
- Implementing refresh tokens






