
import("./env.mjs");

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  trailingSlash: false,
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'avatars.githubusercontent.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'lh3.googleusercontent.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'blatamcursos.s3.amazonaws.com',
        port: '',
        pathname: '/**',
      },
    ],
    domains: ['avatars.githubusercontent.com', 'lh3.googleusercontent.com', 'images.unsplash.com', 'blatamcursos.s3.amazonaws.com'],
  },
  
  serverExternalPackages: [
    '@prisma/client',
    'ioredis',
    'konva', 
    'react-konva', 
    'howler', 
    'tsparticles', 
    'react-particles',
    'three',
    'cannon'
  ],
  
  experimental: {
    optimizePackageImports: ['framer-motion'],
  },
  
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        canvas: false,
        fs: false,
        path: false,
        crypto: false,
      };
    }

    if (isServer) {
      config.externals = config.externals || [];
      config.externals.push(
        'konva',
        'react-konva', 
        'howler',
        'tsparticles',
        'react-particles',
        'recharts',
        'framer-motion',
        'three',
        'cannon'
      );
      
      config.resolve.alias = {
        ...config.resolve.alias,
        'framer-motion': false,
      };
    }

    return config;
  },
};

module.exports = nextConfig;
