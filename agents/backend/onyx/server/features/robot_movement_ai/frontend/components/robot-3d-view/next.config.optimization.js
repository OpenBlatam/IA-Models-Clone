/**
 * Next.js optimization configuration for Robot 3D View
 * 
 * Add these settings to your next.config.js for optimal bundle size
 * 
 * @example
 * const robot3DViewOptimizations = require('./components/robot-3d-view/next.config.optimization');
 * 
 * module.exports = {
 *   ...robot3DViewOptimizations,
 *   // your other config
 * };
 */

module.exports = {
  // Optimize bundle size
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Optimize Three.js imports
      config.resolve.alias = {
        ...config.resolve.alias,
        // Force specific Three.js builds
        'three': require.resolve('three'),
      };

      // Optimize @react-three/drei
      config.resolve.alias = {
        ...config.resolve.alias,
        '@react-three/drei': require.resolve('@react-three/drei'),
      };

      // Tree-shaking optimization
      config.optimization = {
        ...config.optimization,
        usedExports: true,
        sideEffects: false,
      };
    }

    return config;
  },

  // Experimental features for better optimization
  experimental: {
    optimizePackageImports: [
      'three',
      '@react-three/fiber',
      '@react-three/drei',
      '@react-spring/web',
    ],
  },

  // Compression
  compress: true,

  // Production optimizations
  swcMinify: true,

  // Bundle analyzer (install @next/bundle-analyzer)
  // Uncomment to analyze bundle size
  // ...(process.env.ANALYZE === 'true' && {
  //   webpack: (config) => {
  //     const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
  //     config.plugins.push(
  //       new BundleAnalyzerPlugin({
  //         analyzerMode: 'static',
  //         openAnalyzer: false,
  //       })
  //     );
  //     return config;
  //   },
  // }),
};



