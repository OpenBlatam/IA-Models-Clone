# AI Job Replacement Helper - Mobile App

A comprehensive React Native mobile application built with Expo (new architecture) and TypeScript that provides all the functionality of the AI Job Replacement Helper backend API.

## 🚀 Features

### Core Features
- ✅ **Authentication** - Secure login and registration
- ✅ **Gamification System** - Points, levels, badges, and streaks
- ✅ **Job Search (Tinder-style)** - Swipe through jobs with gesture support
- ✅ **Career Roadmap** - Step-by-step guided career transition
- ✅ **Dashboard** - Comprehensive overview of progress and statistics
- ✅ **AI Mentoring** - Chat with AI career coaches
- ✅ **CV Analyzer** - AI-powered resume analysis
- ✅ **Interview Simulator** - Practice interviews with AI feedback
- ✅ **Challenges** - Daily and weekly challenges
- ✅ **Recommendations** - Personalized skill and job recommendations
- ✅ **Notifications** - Real-time notifications system
- ✅ **Content Generator** - AI-powered cover letters, LinkedIn posts, and more
- ✅ **Job Alerts** - Smart job alerts with matching

### Technical Features
- 🎯 **Expo New Architecture** - Latest Expo SDK with new architecture enabled
- 📱 **TypeScript** - Full type safety with strict mode
- 🎨 **Modern UI** - Beautiful, responsive design with gesture animations
- 🔒 **Secure Storage** - Encrypted storage for sensitive data
- 🔄 **React Query** - Efficient data fetching and caching
- 📊 **State Management** - Zustand for global state
- 🧭 **Expo Router** - File-based routing
- 🌙 **Dark Mode Support** - Automatic theme switching with complete theme system
- ♿ **Accessibility** - Full WCAG compliant accessibility support
- ⚡ **Performance Optimized** - Memoization, lazy loading, and optimized renders
- 🎨 **Component Library** - Reusable UI components with consistent styling
- 🌍 **i18n Ready** - Internationalization support (English/Spanish)
- 📐 **Responsive Design** - Adaptive layouts for all screen sizes
- ✅ **Form Validation** - Zod-based validation with custom useForm hook
- 🛡️ **Error Boundaries** - Comprehensive error handling

## 📋 Prerequisites

- Node.js 18+ and npm/yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (for Mac) or Android Emulator
- Backend API running (default: `http://localhost:8030`)

## 🛠️ Installation

1. **Navigate to the project directory:**
```bash
cd agents/backend/onyx/server/features/ai_job_replacement_helper_mobile
```

2. **Install dependencies:**
```bash
npm install
# or
yarn install
```

3. **Configure API endpoint (optional):**
   - The app defaults to `http://localhost:8030`
   - To change it, update `API_BASE_URL` in `src/constants/config.ts`
   - Or set environment variable in `app.json` under `extra.apiUrl`

4. **Start the development server:**
```bash
npm start
# or
expo start
```

5. **Run on your device:**
   - Press `i` for iOS simulator
   - Press `a` for Android emulator
   - Scan QR code with Expo Go app on your physical device

## 📱 Project Structure

```
ai_job_replacement_helper_mobile/
├── app/                    # Expo Router pages
│   ├── (auth)/            # Authentication screens
│   │   ├── login.tsx
│   │   └── register.tsx
│   ├── (tabs)/            # Main app tabs
│   │   ├── dashboard.tsx
│   │   ├── jobs.tsx
│   │   ├── roadmap.tsx
│   │   └── profile.tsx
│   ├── _layout.tsx        # Root layout
│   └── index.tsx          # Entry point
├── src/
│   ├── modules/           # Feature modules (modular architecture)
│   │   ├── auth/          # Authentication module
│   │   │   ├── types.ts
│   │   │   ├── constants.ts
│   │   │   ├── validators.ts
│   │   │   ├── services/
│   │   │   └── index.ts
│   │   ├── jobs/          # Jobs module
│   │   │   ├── types.ts
│   │   │   ├── constants.ts
│   │   │   ├── services/
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   └── index.ts
│   │   ├── gamification/  # Gamification module
│   │   ├── dashboard/     # Dashboard module
│   │   ├── roadmap/       # Roadmap module
│   │   ├── notifications/ # Notifications module
│   │   ├── shared/        # Shared utilities
│   │   └── index.ts      # All modules export
│   ├── components/        # Reusable UI components
│   │   ├── ui/            # Base UI components
│   │   └── dashboard/     # Dashboard components
│   ├── constants/         # App constants
│   │   └── config.ts      # API endpoints and config
│   ├── services/          # API services
│   │   └── api.ts         # Main API service
│   ├── store/             # State management
│   │   ├── authStore.ts   # Authentication state
│   │   └── appStore.ts    # App-wide state
│   ├── theme/             # Theme system
│   │   ├── colors.ts      # Color palettes
│   │   └── theme.ts       # Theme configuration
│   ├── types/             # TypeScript types
│   │   └── index.ts       # All type definitions
│   ├── utils/             # Utility functions
│   │   ├── validation.ts  # Zod schemas
│   │   ├── format.ts      # Formatting helpers
│   │   └── i18n.ts        # Internationalization
│   └── hooks/             # Custom React hooks
│       ├── useApi.ts      # API hooks
│       ├── useForm.ts     # Form handling hook
│       └── useResponsive.ts # Responsive design hook
├── assets/                # Images, fonts, etc.
├── app.json               # Expo configuration
├── package.json           # Dependencies
└── tsconfig.json          # TypeScript config
```

## 🔌 API Integration

The app is fully integrated with the backend API. All endpoints are configured in `src/constants/config.ts` and match the backend API structure:

- **Base URL**: `http://localhost:8030` (configurable)
- **API Version**: `v1`
- **Authentication**: Bearer token (session ID)

### Available API Services

- `apiService.login()` - User authentication
- `apiService.getDashboard()` - Dashboard data
- `apiService.searchJobs()` - Job search
- `apiService.swipeJob()` - Like/dislike/save jobs
- `apiService.getRoadmap()` - Career roadmap
- `apiService.getGamificationProgress()` - User progress
- `apiService.analyzeCV()` - CV analysis
- `apiService.startInterview()` - Interview simulation
- And many more...

See `src/services/api.ts` for the complete API service implementation.

## 🎨 UI Components

The app uses:
- **React Native** core components
- **Expo Vector Icons** for icons
- **React Native Gesture Handler** for swipe gestures
- **React Native Reanimated** for smooth animations
- **React Native Safe Area Context** for safe area handling

## 🔐 Security

- **Encrypted Storage**: Sensitive data stored using `react-native-encrypted-storage`
- **Secure Authentication**: Session-based auth with token storage
- **Input Validation**: Zod schemas for form validation
- **HTTPS**: All API calls should use HTTPS in production

## 📦 Building for Production

### iOS
```bash
expo build:ios
```

### Android
```bash
expo build:android
```

### EAS Build (Recommended)
```bash
npm install -g eas-cli
eas build --platform ios
eas build --platform android
```

## 🧪 Testing

```bash
# Type checking
npm run type-check

# Linting
npm run lint
```

## 🐛 Troubleshooting

### API Connection Issues
- Ensure backend is running on `http://localhost:8030`
- For physical devices, use your computer's IP address instead of `localhost`
- Check CORS settings on the backend

### Build Issues
- Clear cache: `expo start -c`
- Reinstall dependencies: `rm -rf node_modules && npm install`
- Check Expo SDK version compatibility

## 📚 Documentation

- [Expo Documentation](https://docs.expo.dev/)
- [React Native Documentation](https://reactnative.dev/)
- [Expo Router Documentation](https://docs.expo.dev/router/introduction/)
- [React Query Documentation](https://tanstack.com/query/latest)
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Detailed list of improvements and new features
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [ASSETS.md](ASSETS.md) - Asset requirements
- [LIBRARIES.md](LIBRARIES.md) - Complete library list and documentation
- [BEST_LIBRARIES.md](BEST_LIBRARIES.md) - Best practices for choosing libraries
- [HOOKS_GUIDE.md](src/hooks/HOOKS_GUIDE.md) - Complete hooks documentation
- [MODULAR_STRUCTURE.md](MODULAR_STRUCTURE.md) - Modular architecture guide
- [MODULAR_BENEFITS.md](MODULAR_BENEFITS.md) - Benefits of modular architecture

## 🔄 Backend Compatibility

This mobile app is designed to work seamlessly with the AI Job Replacement Helper backend API. All endpoints and data structures match the backend implementation.

**Backend Location**: `agents/backend/onyx/server/features/ai_job_replacement_helper`

## 🪝 Custom Hooks Library

The app includes **25+ custom hooks** for common React Native patterns:

### Core Hooks
- `useForm` - Form handling with Zod validation
- `useAuthenticatedQuery` - React Query with auth
- `useAuthenticatedMutation` - Mutations with auth
- `useResponsive` - Responsive design breakpoints

### Utility Hooks
- `useDebounce` - Debounce values
- `useThrottle` - Throttle functions
- `usePrevious` - Get previous value
- `useToggle` - Toggle boolean state
- `useAsync` - Handle async operations

### Storage Hooks
- `useSecureStorage` - Encrypted storage

### Network & Status Hooks
- `useNetworkStatus` - Monitor connectivity
- `useAppState` - Monitor app state

### UI Hooks
- `useKeyboard` - Keyboard visibility
- `useFocus` - Screen focus detection
- `useAnimation` - Animation control
- `useOrientation` - Device orientation
- `useDimensions` - Screen dimensions
- `useSafeArea` - Safe area insets

### Timer Hooks
- `useInterval` - Execute at intervals
- `useTimeout` - Execute after delay

### Platform Hooks
- `usePermissions` - Request permissions
- `useImagePicker` - Pick images
- `useDeepLink` - Handle deep links
- `useBackHandler` - Android back button
- `useClipboard` - Clipboard operations
- `useVibration` - Haptic feedback
- `useLocation` - Get device location
- `useBiometrics` - Biometric authentication

See [HOOKS_GUIDE.md](src/hooks/HOOKS_GUIDE.md) for complete documentation and examples.

## 🎨 Component Library

The app includes a comprehensive component library:

- **Button**: Multiple variants, sizes, and states
- **Input**: Form inputs with validation and icons
- **Card**: Container component with shadows and padding
- **Loading**: Loading states with messages
- **EmptyState**: Empty state screens with actions
- **ErrorBoundary**: Global error handling

All components are:
- Fully typed with TypeScript
- Memoized for performance
- Theme-aware (light/dark mode)
- Accessibility compliant
- Responsive

## ⚡ Performance

The app is optimized for performance:
- Component memoization with `React.memo`
- Expensive calculations memoized with `useMemo`
- Event handlers optimized with `useCallback`
- React Query caching configured
- Lazy loading structure ready
- Optimized re-renders

## 🌍 Internationalization

The app supports multiple languages:
- English (default)
- Spanish
- Easy to extend to more languages

See `src/utils/i18n.ts` for translation structure.

## 📦 Libraries Used

This project uses industry-standard libraries:

### Core
- **@tanstack/react-query** - Server state & caching
- **zustand** - Global state management
- **zod** - Schema validation
- **react-hook-form** - Form handling

### UI & Animations
- **react-native-paper** - Material Design components
- **react-native-reanimated** - High-performance animations
- **react-native-gesture-handler** - Gesture recognition
- **react-native-modal** - Modal component
- **react-native-bottom-sheet** - Bottom sheets
- **react-native-skeleton-placeholder** - Loading skeletons

### Platform Features
- **expo-notifications** - Push notifications
- **expo-image-picker** - Image/video picker
- **expo-camera** - Camera access
- **expo-location** - Location services
- **expo-local-authentication** - Biometric auth
- **expo-file-system** - File operations
- **expo-sharing** - Share content
- **expo-contacts** - Contacts access
- **expo-calendar** - Calendar access

### Utilities
- **date-fns** - Date utilities
- **lodash** - Utility functions
- **immer** - Immutable updates
- **axios** - HTTP client

### Testing
- **jest** - Unit testing
- **@testing-library/react-native** - Testing utilities
- **detox** - E2E testing

See [LIBRARIES.md](LIBRARIES.md) and [BEST_LIBRARIES.md](BEST_LIBRARIES.md) for complete documentation.

## 🎯 Best Practices

This app follows all React Native and Expo best practices:
- ✅ Functional components only
- ✅ TypeScript strict mode
- ✅ Proper error handling
- ✅ Accessibility standards (WCAG)
- ✅ Performance optimization
- ✅ Code reusability
- ✅ Clean architecture
- ✅ Consistent styling
- ✅ Responsive design
- ✅ Industry-standard libraries
- ✅ Comprehensive testing setup
- ✅ **Modular Architecture** - Feature-based modules with clear separation
- ✅ **Service Layer** - Business logic separated from UI
- ✅ **Type Safety** - Complete TypeScript coverage
- ✅ **Reusable Components** - Modular, composable components

## 📝 License

Proprietary - Blatam Academy

## 👥 Author

Blatam Academy

---

**Version**: 1.0.0  
**Last Updated**: 2024

