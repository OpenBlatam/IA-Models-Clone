# Setup Guide - Music Analyzer AI Mobile App

## Prerequisites

- Node.js 18+ and npm/yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (for Mac) or Android Studio (for Android development)
- Backend API running on port 8010 (or configure in `app.json`)

## Installation Steps

1. **Navigate to the mobile app directory:**
```bash
cd mobile_app
```

2. **Install dependencies:**
```bash
npm install
# or
yarn install
```

3. **Configure API endpoint:**
   - Edit `app.json` and update the `extra.apiUrl` field with your backend URL
   - Default: `http://localhost:8010`

4. **Start the development server:**
```bash
npm start
# or
yarn start
```

5. **Run on your preferred platform:**
   - Press `i` for iOS simulator
   - Press `a` for Android emulator
   - Scan QR code with Expo Go app on your physical device

## Project Structure

```
mobile_app/
├── src/
│   ├── app/                    # Expo Router pages
│   │   ├── _layout.tsx        # Root layout with providers
│   │   ├── index.tsx          # Home screen
│   │   ├── search.tsx         # Search page
│   │   └── analysis.tsx       # Analysis page
│   ├── components/
│   │   ├── common/            # Reusable components
│   │   │   ├── error-boundary.tsx
│   │   │   ├── error-message.tsx
│   │   │   ├── loading-spinner.tsx
│   │   │   └── empty-state.tsx
│   │   └── music/             # Music-specific components
│   │       ├── track-card.tsx
│   │       ├── search-screen.tsx
│   │       ├── analysis-screen.tsx
│   │       └── visualization-card.tsx
│   ├── contexts/
│   │   └── music-context.tsx  # Global music state
│   ├── hooks/
│   │   ├── use-music-analysis.ts
│   │   └── use-debounce.ts
│   ├── services/
│   │   ├── api-client.ts      # Axios client
│   │   └── music-api.ts       # Music API service
│   ├── types/
│   │   └── api.ts             # TypeScript interfaces
│   ├── utils/
│   │   ├── formatters.ts
│   │   ├── validation.ts      # Zod schemas
│   │   └── storage.ts         # AsyncStorage helpers
│   └── constants/
│       └── config.ts          # App constants
├── assets/                    # Images, fonts (create these)
├── app.json                   # Expo configuration
├── package.json
├── tsconfig.json
└── babel.config.js
```

## Key Features Implemented

### ✅ Core Functionality
- [x] Search tracks via Spotify API
- [x] Display track analysis (key, tempo, technical features)
- [x] Show coaching recommendations
- [x] Recent searches history
- [x] Favorites management
- [x] Error handling and retry
- [x] Loading states
- [x] Empty states

### ✅ Technical Implementation
- [x] TypeScript with strict mode
- [x] React Query for data fetching
- [x] React Context for global state
- [x] Zod validation
- [x] Error boundaries
- [x] Safe area handling
- [x] Dark mode support
- [x] Debounced search
- [x] Type-safe navigation

### ✅ Best Practices
- [x] Functional components
- [x] Custom hooks
- [x] Component composition
- [x] Proper error handling
- [x] Performance optimizations
- [x] Code organization
- [x] Type safety

## Next Steps

1. **Create assets:**
   - Add `icon.png` (1024x1024)
   - Add `splash.png` (1242x2436)
   - Add `adaptive-icon.png` (1024x1024)
   - Add `favicon.png` (48x48)

2. **Configure environment:**
   - Set up production API URL
   - Configure app signing for iOS/Android
   - Set up OTA updates with Expo Updates

3. **Testing:**
   - Write unit tests for utilities
   - Write integration tests for flows
   - Test on physical devices

4. **Enhancements:**
   - Add favorites screen
   - Add history screen
   - Add recommendations screen
   - Add sharing functionality
   - Add offline support
   - Add push notifications

## Troubleshooting

### Common Issues

1. **API connection errors:**
   - Verify backend is running
   - Check API URL in `app.json`
   - Verify CORS settings on backend

2. **TypeScript errors:**
   - Run `npm run type-check`
   - Ensure all dependencies are installed

3. **Build errors:**
   - Clear cache: `expo start -c`
   - Delete `node_modules` and reinstall
   - Check Expo SDK version compatibility

## Development Commands

```bash
# Start development server
npm start

# Type checking
npm run type-check

# Linting
npm run lint

# Run tests
npm test

# Build for production
expo build:ios
expo build:android
```

## Resources

- [Expo Documentation](https://docs.expo.dev/)
- [React Native Documentation](https://reactnative.dev/)
- [Expo Router Documentation](https://docs.expo.dev/router/introduction/)
- [React Query Documentation](https://tanstack.com/query/latest)

