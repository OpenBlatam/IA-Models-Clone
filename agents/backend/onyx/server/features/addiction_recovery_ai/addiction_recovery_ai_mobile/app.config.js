module.exports = {
  expo: {
    name: 'Addiction Recovery AI',
    slug: 'addiction-recovery-ai',
    version: '1.0.0',
    orientation: 'portrait',
    icon: './assets/icon.png',
    userInterfaceStyle: 'automatic',
    splash: {
      image: './assets/splash.png',
      resizeMode: 'contain',
      backgroundColor: '#ffffff',
    },
    assetBundlePatterns: ['**/*'],
    ios: {
      supportsTablet: true,
      bundleIdentifier: 'com.addictionrecoveryai.app',
      buildNumber: '1.0.0',
    },
    android: {
      adaptiveIcon: {
        foregroundImage: './assets/adaptive-icon.png',
        backgroundColor: '#ffffff',
      },
      package: 'com.addictionrecoveryai.app',
      versionCode: 1,
    },
    web: {
      favicon: './assets/favicon.png',
    },
    plugins: [
      'expo-router',
      [
        'expo-build-properties',
        {
          android: {
            enableProguardInReleaseBuilds: true,
            proguardFiles: ['proguard-rules.pro'],
          },
          ios: {
            deploymentTarget: '13.0',
          },
        },
      ],
    ],
    extra: {
      apiBaseUrl: process.env.API_BASE_URL || 'http://localhost:8018',
      apiPrefix: process.env.API_PREFIX || '/recovery',
    },
    experiments: {
      typedRoutes: true,
    },
    // Performance optimizations
    jsEngine: 'hermes',
    updates: {
      fallbackToCacheTimeout: 0,
    },
  },
};
