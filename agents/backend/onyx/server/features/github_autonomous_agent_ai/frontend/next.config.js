/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Turbopack está habilitado por defecto en Next.js 15 con --turbo flag
  // No requiere configuración adicional en next.config.js

  // Configuración para API routes sin límites de timeout
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },

  // Configuración para evitar timeouts en desarrollo
  onDemandEntries: {
    // Período en ms que una página debe estar inactiva antes de ser descartada
    maxInactiveAge: 60 * 60 * 1000, // 1 hora
    // Número de páginas que deben mantenerse simultáneamente sin ser descartadas
    pagesBufferLength: 5,
  },

  // Proxy requests al backend unificado para evitar CORS
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8030';
    return [
      {
        source: '/backend-api/:path*',
        destination: `${backendUrl}/api/v1/:path*`,
      },
      {
        source: '/backend-ws/:path*',
        destination: `${backendUrl}/ws/:path*`,
      },
      {
        source: '/backend-health',
        destination: `${backendUrl}/health`,
      },
    ];
  },
}

module.exports = nextConfig


