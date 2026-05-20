# Etapa de build
FROM node:20-alpine AS builder
WORKDIR /app

# Install system dependencies for canvas and other native modules
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    cairo-dev \
    jpeg-dev \
    pango-dev \
    musl-dev \
    giflib-dev \
    pixman-dev \
    pangomm-dev \
    libjpeg-turbo-dev \
    freetype-dev \
    openssl-dev \
    openssl

# Copia todo el código fuente primero
COPY . .

# Instala dependencias con resolución forzada para Linux
RUN npm install --no-package-lock --legacy-peer-deps --omit=optional --ignore-scripts --force

# Set environment variables for build (these will be overridden at runtime)
ENV NODE_ENV=production
ENV SKIP_ENV_VALIDATION=true
ENV DATABASE_URL="postgresql://placeholder"
ENV AUTH_SECRET="placeholder"
ENV GOOGLE_CLIENT_ID="placeholder"
ENV GOOGLE_CLIENT_SECRET="placeholder"
ENV AWS_ACCESS_KEY_ID="placeholder"
ENV AWS_SECRET_ACCESS_KEY="placeholder"
ENV OPENAI_API_KEY="placeholder"
ENV NEXT_PUBLIC_SITE_URL="https://blatam-academy.us-east-1.elasticbeanstalk.com"
ENV NEXT_PUBLIC_CDN_URL="https://d1xxxxxxxx.cloudfront.net"
ENV NEXT_PUBLIC_APP_URL="https://blatam-academy.us-east-1.elasticbeanstalk.com"

# Generate Prisma client for Alpine Linux
RUN npx prisma generate

# Set environment variables for build (these will be overridden at runtime)
ENV NODE_ENV=production
ENV SKIP_ENV_VALIDATION=true
ENV DATABASE_URL="postgresql://placeholder"
ENV AUTH_SECRET="placeholder"
ENV GOOGLE_CLIENT_ID="placeholder"
ENV GOOGLE_CLIENT_SECRET="placeholder"
ENV AWS_ACCESS_KEY_ID="placeholder"
ENV AWS_SECRET_ACCESS_KEY="placeholder"
ENV NEXT_PUBLIC_SITE_URL="https://blatam-academy.us-east-1.elasticbeanstalk.com"
ENV NEXT_PUBLIC_CDN_URL="https://d1xxxxxxxx.cloudfront.net"
ENV NEXT_PUBLIC_APP_URL="https://blatam-academy.us-east-1.elasticbeanstalk.com"

# Generate Prisma client for Alpine Linux
RUN npx prisma generate

# Construye la aplicación
RUN npm run build

# Etapa de producción
FROM node:20-alpine AS runner
WORKDIR /app

# Install runtime dependencies for Prisma and health checks
RUN apk add --no-cache openssl curl

COPY --from=builder /app/package.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/prisma ./prisma

EXPOSE 3000
CMD ["npm", "start"]   