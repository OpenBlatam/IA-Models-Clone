export default {
  expo: {
    name: 'AI Project Generator',
    slug: 'ai-project-generator',
    version: '1.0.0',
    orientation: 'portrait',
    icon: './assets/icon.png',
    userInterfaceStyle: 'light',
    splash: {
      image: './assets/splash.png',
      resizeMode: 'contain',
      backgroundColor: '#ffffff',
    },
    assetBundlePatterns: ['**/*'],
    ios: {
      supportsTablet: true,
      bundleIdentifier: 'com.blatamacademy.aiprojectgenerator',
    },
    android: {
      adaptiveIcon: {
        foregroundImage: './assets/adaptive-icon.png',
        backgroundColor: '#ffffff',
      },
      package: 'com.blatamacademy.aiprojectgenerator',
    },
    web: {
      favicon: './assets/favicon.png',
    },
    scheme: 'ai-project-generator',
    extra: {
      apiUrl: process.env.API_URL || 'http://localhost:8020',
    },
  },
};

