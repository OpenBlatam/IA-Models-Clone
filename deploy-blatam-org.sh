#!/bin/bash
set -e

echo "🚀 Starting Blatam.org Production Deployment..."

echo "📦 Building application..."
npm run build

if [ $? -ne 0 ]; then
  echo "❌ Build failed. Deployment aborted."
  exit 1
fi

echo "🐳 Building Docker image..."
docker build -t blatam-academy:latest .

echo "🛑 Stopping existing container..."
ssh -o StrictHostKeyChecking=no -i ~/attachments/66f1bcb0-2197-4a47-83ce-9b53fa241bbd/blatam.pem ubuntu@18.206.225.74 "docker stop blatam-academy || true && docker rm blatam-academy || true"

echo "📤 Transferring image to EC2..."
docker save blatam-academy:latest | gzip > blatam-academy-latest.tar.gz
scp -o StrictHostKeyChecking=no -i ~/attachments/66f1bcb0-2197-4a47-83ce-9b53fa241bbd/blatam.pem blatam-academy-latest.tar.gz ubuntu@18.206.225.74:~/

echo "📋 Transferring environment configuration..."
scp -o StrictHostKeyChecking=no -i ~/attachments/66f1bcb0-2197-4a47-83ce-9b53fa241bbd/blatam.pem .env ubuntu@18.206.225.74:~/

echo "🚀 Deploying on EC2..."
ssh -o StrictHostKeyChecking=no -i ~/attachments/66f1bcb0-2197-4a47-83ce-9b53fa241bbd/blatam.pem ubuntu@18.206.225.74 << 'EOF'
echo "Loading Docker image..."
docker load < blatam-academy-latest.tar.gz

echo "Starting Blatam Academy container..."
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

echo "Configuring port forwarding..."
sudo iptables -t nat -C PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 3000 2>/dev/null || \
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 3000

sudo mkdir -p /etc/iptables
sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null

echo "⏳ Waiting for application to be healthy..."
sleep 45

if docker ps | grep -q blatam-academy; then
  echo "✅ Container is running"
  
  for i in {1..10}; do
    if curl -f http://localhost:3000/api/health; then
      echo "✅ Health check passed"
      break
    else
      echo "⏳ Health check attempt $i/10 failed, retrying..."
      sleep 5
    fi
  done
  
  if curl -f http://localhost:3000/ | grep -q "Blatam Academy"; then
    echo "✅ Application is serving correctly"
  else
    echo "❌ Application not serving expected content"
    exit 1
  fi
else
  echo "❌ Container failed to start"
  docker logs blatam-academy || true
  exit 1
fi
EOF

echo "🔍 Final verification from external..."
sleep 10

for i in {1..5}; do
  if curl -f http://blatam.org/ | grep -q "Blatam Academy"; then
    echo "✅ blatam.org is serving Blatam Academy application successfully!"
    break
  else
    echo "⏳ External verification attempt $i/5 failed, retrying..."
    sleep 10
  fi
done

if curl -f http://blatam.org/api/health; then
  echo "✅ External health check passed"
else
  echo "❌ External health check failed"
fi

echo "🎉 Deployment completed successfully!"
echo "🌐 Blatam Academy is now live at http://blatam.org"
