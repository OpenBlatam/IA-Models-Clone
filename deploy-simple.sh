#!/bin/bash
cd /home/ec2-user
rm -rf blatam-academy
mkdir -p blatam-academy
cd blatam-academy

# Install Node.js 20 if not available
curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
sudo yum install -y nodejs

# Create a simple Next.js app structure
npm init -y
npm install next@latest react@latest react-dom@latest

# Create basic app structure
mkdir -p app pages public
echo 'export default function Home() { return <div><h1>Blatam Academy</h1><p>Aplicación desplegada exitosamente en EC2</p></div>; }' > app/page.js
echo '{ "scripts": { "dev": "next dev", "build": "next build", "start": "next start -p 3000" } }' > package.json

# Build and start the app
npm run build
nohup npm start > app.log 2>&1 &

echo 'Deployment completed. App running on port 3000'

