const { withContentlayer } = require("next-contentlayer2");

import("./env.mjs");

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  trailingSlash: false,
  poweredByHeader: false,
  compress: true,

  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "avatars.githubusercontent.com",
      },
      {
        protocol: "https",
        hostname: "lh3.googleusercontent.com",
      },
      {
        protocol: "https",
        hostname: "randomuser.me",
      },
      {
        protocol: "https",
        hostname: "images.unsplash.com",
      },
      {
        protocol: "https",
        hostname: "example.com",
      },
      {
        protocol: "https",
        hostname: "blatamcursos.s3.amazonaws.com",
      },
    ],
    domains: ['avatars.githubusercontent.com', 'lh3.googleusercontent.com', 'images.unsplash.com', 'example.com', 'blatamcursos.s3.amazonaws.com'],
    formats: ['image/webp', 'image/avif'],
    minimumCacheTTL: 60,
    dangerouslyAllowSVG: true,
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },
  experimental: {
    optimizeCss: false,
    serverComponentsExternalPackages: ['konva', 'react-konva', 'howler', 'tsparticles'],
    esmExternals: 'loose',
  },
  serverExternalPackages: [
    '@prisma/client', 
    'ioredis', 
    'konva', 
    'react-konva', 
    'howler', 
    'tsparticles', 
    'react-particles',
    'chart.js',
    'react-chartjs-2',
    'three',
    'cannon',
    'cannon-es',
    'react-three-fiber',
    '@react-three/fiber',
    '@react-three/drei'
  ],
  turbopack: {
    rules: {
      '*.svg': {
        loaders: ['@svgr/webpack'],
        as: '*.js',
      },
    },
  },
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production' ? {
      exclude: ['error', 'warn']
    } : false,
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      const originalEntry = config.entry;
      config.entry = async () => {
        const entries = await originalEntry();
        
        Object.keys(entries).forEach(key => {
          if (Array.isArray(entries[key])) {
            if (!entries[key].includes('./polyfills.js')) {
              entries[key].unshift('./polyfills.js');
            }
          }
        });
        
        return entries;
      };
    }

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
      config.resolve.fallback = {
        ...config.resolve.fallback,
        canvas: false,
        encoding: false,
        'react-konva': false,
        'konva': false,
        'howler': false,
        'tsparticles': false,
        'react-particles': false,
        'self': false,
        'window': false,
        'document': false,
        'navigator': false,
      };
    }

    config.resolve.alias = {
      ...config.resolve.alias,
      'react-konva': 'react-konva/lib/ReactKonva',
      'konva': 'konva/lib/index-umd',
    };

    if (isServer) {
      config.externals = config.externals || [];
      
      const originalExternals = config.externals;
      config.externals = [
        ...originalExternals,
        function({context, request}, callback) {
          const problematicPackages = [
            'recharts', 'react-transition-group', 'thenify', 'howler', 
            'konva', 'react-konva', 'tsparticles', 'react-particles',
            'chart.js', 'react-chartjs-2', 'framer-motion', 'three',
            'cannon', 'cannon-es', 'react-three-fiber', '@react-three',
            'gsap', '@react-spring', 'lottie-react', 'react-lottie'
          ];
          
          if (request && (
            problematicPackages.some(pkg => request.includes(pkg)) ||
            request.includes('self') ||
            request.includes('window') ||
            request.includes('document')
          )) {
            return callback(null, `commonjs ${request}`);
          }
          
          callback();
        }
      ];
    }

    if (!isServer) {
      config.optimization = {
        ...config.optimization,
        splitChunks: {
          chunks: 'all',
          minSize: 20000,
          maxSize: 244000,
          cacheGroups: {
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: 'vendors',
              chunks: 'all',
              priority: 10,
            },
            ui: {
              test: /[\\/]node_modules[\\/](@radix-ui|@headlessui)[\\/]/,
              name: 'ui',
              chunks: 'all',
              priority: 15,
            },
          },
        },
      };
    } else {
      config.optimization = {
        ...config.optimization,
        splitChunks: false,
      };
    }

    return config;
  },
};

module.exports = withContentlayer(nextConfig);
