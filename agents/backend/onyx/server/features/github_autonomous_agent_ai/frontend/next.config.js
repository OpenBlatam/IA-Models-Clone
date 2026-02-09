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
}

module.exports = nextConfig


