#!/bin/bash

# CONFIGURA ESTAS VARIABLES:
EC2_USER=ec2-user
EC2_HOST=ec2-18-206-225-74.compute-1.amazonaws.com
KEY_PATH=~/Downloads/blatam.pem
PROJECT_DIR=next-saas-stripe-starter 

# 1. Empaqueta tu código (opcional si no usas git push)
tar czf $PROJECT_DIR.tar.gz $PROJECT_DIR

# 2. Copia el proyecto a la EC2
scp -i $KEY_PATH $PROJECT_DIR.tar.gz $EC2_USER@$EC2_HOST:~/

# 3. Conéctate y ejecuta los comandos de despliegue
ssh -i $KEY_PATH $EC2_USER@$EC2_HOST << 'ENDSSH'
  set -e
  # Instala Docker si no está instalado
  if ! command -v docker &> /dev/null; then
    sudo yum update -y || sudo apt-get update -y
    sudo yum install docker git -y || sudo apt-get install docker.io git -y
    sudo service docker start || sudo systemctl start docker
    sudo usermod -aG docker $USER
    newgrp docker
  fi

  # Descomprime el proyecto
  tar xzf next-saas-stripe-starter.tar.gz
  cd next-saas-stripe-starter

  # Construye la imagen Docker
  docker build --platform=linux/arm64 -t my-nextjs-app .

  # Detén y elimina el contenedor anterior si existe
  docker stop my-nextjs-app || true
  docker rm my-nextjs-app || true

  # Ejecuta el contenedor
  docker run -d -p 3000:3000 --name my-nextjs-app my-nextjs-app
ENDSSH

echo "¡Despliegue completado! Accede a http://$EC2_HOST:3000" 