# AWS Deployment README
# Place this file in the ssl/ directory for reference

## SSL Certificate Setup

### Option 1: Let's Encrypt (Free, Recommended for Production)

1. Install certbot on your EC2:
   ```bash
   sudo yum install -y certbot  # Amazon Linux
   # or
   sudo apt-get install -y certbot  # Ubuntu
   ```

2. Stop nginx temporarily:
   ```bash
   docker-compose -f aws/docker-compose.yml stop nginx
   ```

3. Get certificate:
   ```bash
   sudo certbot certonly --standalone -d your-domain.com
   ```

4. Copy certificates:
   ```bash
   sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/
   sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/
   ```

5. Update nginx.conf to use the certificates (uncomment ssl_certificate lines)

6. Restart nginx:
   ```bash
   docker-compose -f aws/docker-compose.yml up -d nginx
   ```

### Option 2: Self-Signed (Development Only)

```bash
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/privkey.pem \
    -out ssl/fullchain.pem \
    -subj "/CN=localhost"
```

### Option 3: AWS Certificate Manager

If using an ALB (Application Load Balancer), you can use ACM for free SSL certificates:

1. Request a certificate in AWS ACM
2. Validate via DNS or email
3. Attach to your ALB listener
4. Point nginx to use HTTP internally (remove SSL config from nginx)
