#!/bin/bash

# Construir la imagen
echo "Construyendo imagen Docker..."
docker build -t blatam-academy .

# Ejecutar el contenedor
echo "Iniciando contenedor..."
docker run -p 3000:3000 \
  -e DATABASE_URL="postgresql://neondb_owner:npg_ejg2Yr1EZMOw@ep-dark-sky-a5c88yly-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require" \
  -e AUTH_SECRET="32mwUMnlOtkH1H8irIqxlaL80KvgdjUlbg2IovnhL1s=" \
  -e GOOGLE_CLIENT_ID="766657034501-2fcprad0507agtsvn5gvgvpv0ggqm80i.apps.googleusercontent.com" \
  -e GOOGLE_CLIENT_SECRET="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" \
  -e NEXT_PUBLIC_SITE_URL="http://localhost:3000" \
  blatam-academy 