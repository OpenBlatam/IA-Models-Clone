#!/bin/bash
set -e

echo "🚀 Starting Blatam Academy Production Deployment..."

echo "📦 Building application..."
npm run build

if [ $? -ne 0 ]; then
  echo "❌ Build failed. Deployment aborted."
  exit 1
fi

echo "🐳 Building Docker image..."
docker build -t blatam-academy:latest .

echo "🛑 Stopping existing container..."
ssh -i ~/attachments/66f1bcb0-2197-4a47-83ce-9b53fa241bbd/blatam.pem ubuntu@18.206.225.74 "docker stop blatam-academy || true && docker rm blatam-academy || true"

echo "📤 Transferring image to EC2..."
docker save blatam-academy:latest | gzip > blatam-academy-latest.tar.gz
scp -i ~/attachments/66f1bcb0-2197-4a47-83ce-9b53fa241bbd/blatam.pem blatam-academy-latest.tar.gz ubuntu@18.206.225.74:~/

echo "🚀 Deploying on EC2..."
ssh -i ~/attachments/66f1bcb0-2197-4a47-83ce-9b53fa241bbd/blatam.pem ubuntu@18.206.225.74 << 'EOF'
docker load < blatam-academy-latest.tar.gz

docker run -d \
  --name blatam-academy \
  --restart unless-stopped \
  -p 3000:3000 \
  --env-file .env \
  --health-cmd="curl -f http://localhost:3000/api/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  blatam-academy:latest

sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 3000
sudo iptables-save > /etc/iptables/rules.v4

echo "⏳ Waiting for application to be healthy..."
sleep 30

if docker ps | grep -q blatam-academy; then
  echo "✅ Container is running"
  if curl -f http://localhost:3000/api/health; then
    echo "✅ Health check passed"
  else
    echo "❌ Health check failed"
    exit 1
  fi
else
  echo "❌ Container failed to start"
  exit 1
fi
EOF

echo "🔍 Final verification..."
sleep 10
if curl -f http://blatam.org/api/health; then
  echo "✅ Deployment successful! blatam.org is live"
else
  echo "❌ Final verification failed"
  exit 1
fi

echo "🎉 Deployment completed successfully!"
