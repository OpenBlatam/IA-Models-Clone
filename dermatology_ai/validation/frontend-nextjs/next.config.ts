import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: true,
  // Turbopack is enabled by default in Next.js 15.4 when using --turbo flag
  experimental: {
    turbo: {
      // Turbopack configuration
    },
  },
};

export default nextConfig;






