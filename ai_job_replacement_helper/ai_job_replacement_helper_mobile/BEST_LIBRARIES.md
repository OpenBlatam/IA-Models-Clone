# Best Libraries for React Native/Expo - Complete Guide

## 🏆 Top Recommended Libraries

### ⭐ Essential (Must Have)

1. **@tanstack/react-query** - Server state management
   - Why: Best caching, synchronization, and data fetching
   - Alternative: SWR (simpler but less features)

2. **zustand** - Global state management
   - Why: Lightweight, simple API, great TypeScript support
   - Alternative: Redux Toolkit (more features, more boilerplate)

3. **zod** - Schema validation
   - Why: TypeScript-first, runtime validation, great DX
   - Alternative: Yup (more features, less TypeScript-friendly)

4. **react-native-reanimated** - Animations
   - Why: 60fps animations, runs on UI thread
   - Alternative: Animated API (built-in but slower)

5. **react-native-gesture-handler** - Gestures
   - Why: Better gesture recognition, required for Reanimated
   - Alternative: PanResponder (built-in but limited)

6. **expo-router** - Navigation
   - Why: File-based routing, deep linking, type-safe
   - Alternative: React Navigation (more control, more setup)

7. **react-native-safe-area-context** - Safe areas
   - Why: Handle notches, status bars properly
   - Alternative: Manual padding (error-prone)

8. **react-native-encrypted-storage** - Secure storage
   - Why: Encrypted keychain/keystore access
   - Alternative: AsyncStorage (not secure)

### 🎨 UI Libraries

1. **react-native-paper** - Material Design
   - Best for: Material Design apps
   - Bundle size: Medium
   - Customization: High

2. **NativeBase** - Cross-platform components
   - Best for: Customizable design systems
   - Bundle size: Large
   - Customization: Very High

3. **react-native-elements** - UI toolkit
   - Best for: Quick prototyping
   - Bundle size: Medium
   - Customization: Medium

4. **react-native-modal** - Modal component
   - Best for: Overlays and modals
   - Bundle size: Small
   - Customization: High

5. **react-native-bottom-sheet** - Bottom sheets
   - Best for: Bottom sheet UIs
   - Bundle size: Small
   - Customization: High

### 📝 Forms

1. **react-hook-form** ⭐ Recommended
   - Why: Best performance, minimal re-renders
   - Bundle size: Small
   - TypeScript: Excellent

2. **Formik** - Alternative
   - Why: More features, larger community
   - Bundle size: Medium
   - TypeScript: Good

### 🧪 Testing

1. **Jest** - Unit testing
   - Why: Standard, great React Native support
   - Setup: Easy with jest-expo

2. **@testing-library/react-native** - Testing utilities
   - Why: Best practices, accessible queries
   - Alternative: Enzyme (older, less maintained)

3. **Detox** - E2E testing
   - Why: Real device testing, stable
   - Alternative: Appium (more complex)

### 🐛 Error Tracking

1. **@sentry/react-native** ⭐ Recommended
   - Why: Best error tracking, crash reporting
   - Features: Source maps, breadcrumbs, releases
   - Alternative: Bugsnag, Rollbar

### 📊 Analytics

1. **Firebase Analytics** - Google Analytics
   - Why: Free, comprehensive, easy setup
   - Alternative: Mixpanel, Amplitude

2. **expo-analytics** - Expo analytics
   - Why: Simple, Expo-integrated
   - Alternative: Custom solution

### 🔔 Notifications

1. **expo-notifications** - Push notifications
   - Why: Expo-managed, cross-platform
   - Alternative: react-native-push-notification

### 📸 Media

1. **expo-image-picker** - Image/video picker
   - Why: Expo-managed, easy permissions
   - Alternative: react-native-image-picker

2. **expo-camera** - Camera access
   - Why: Expo-managed, good API
   - Alternative: react-native-camera

3. **expo-media-library** - Media library
   - Why: Access device media
   - Alternative: react-native-fs

### 🗺️ Maps

1. **react-native-maps** - Maps
   - Why: Best map library for RN
   - Alternative: Mapbox (paid, more features)

### 📅 Date & Time

1. **date-fns** ⭐ Recommended
   - Why: Lightweight, tree-shakeable, immutable
   - Alternative: moment.js (larger, mutable)

2. **dayjs** - Alternative
   - Why: Smaller than moment, similar API
   - Alternative: date-fns (more functions)

### 🎭 Icons

1. **@expo/vector-icons** - Icon library
   - Why: Multiple icon sets, Expo-integrated
   - Alternative: react-native-vector-icons

### 🔐 Authentication

1. **expo-auth-session** - OAuth
   - Why: Expo-managed, secure
   - Alternative: react-native-app-auth

2. **expo-google-sign-in** - Google Sign-In
   - Why: Easy Google authentication
   - Alternative: Custom OAuth

### 💰 Payments

1. **expo-in-app-purchases** - In-app purchases
   - Why: Expo-managed, cross-platform
   - Alternative: react-native-iap

### 📱 Platform Features

1. **expo-location** - Location services
2. **expo-contacts** - Contacts access
3. **expo-calendar** - Calendar access
4. **expo-file-system** - File operations
5. **expo-sharing** - Share content
6. **expo-clipboard** - Clipboard
7. **expo-haptics** - Haptic feedback
8. **expo-local-authentication** - Biometrics
9. **expo-web-browser** - In-app browser

### 🛠️ Utilities

1. **lodash** - Utility functions
   - Why: Comprehensive utilities
   - Note: Use specific imports to reduce bundle size

2. **immer** - Immutable updates
   - Why: Easier state updates
   - Alternative: Manual immutability

3. **react-native-skeleton-placeholder** - Loading skeletons
   - Why: Better UX than spinners
   - Alternative: Custom skeletons

## 📦 Bundle Size Considerations

### Small Bundle (< 50KB)
- zod
- date-fns
- immer
- react-hook-form

### Medium Bundle (50-200KB)
- zustand
- react-native-paper
- react-native-modal
- @tanstack/react-query

### Large Bundle (> 200KB)
- NativeBase
- react-native-maps
- Firebase SDK

## 🎯 Recommendations by Use Case

### Startup/MVP
- expo-router
- zustand
- @tanstack/react-query
- zod
- react-native-paper
- date-fns

### Enterprise App
- expo-router
- Redux Toolkit (if complex state)
- @tanstack/react-query
- @sentry/react-native
- Firebase Analytics
- Detox

### High Performance App
- react-native-reanimated
- react-native-gesture-handler
- react-hook-form
- immer
- expo-image

### Design-Heavy App
- react-native-paper or NativeBase
- react-native-reanimated
- react-native-bottom-sheet
- react-native-modal
- react-native-skeleton-placeholder

## 🔄 Migration Guide

### From Redux to Zustand
```typescript
// Redux
const count = useSelector(state => state.count);
dispatch(increment());

// Zustand
const count = useStore(state => state.count);
increment();
```

### From Formik to react-hook-form
```typescript
// Formik
<Formik initialValues={{ email: '' }} onSubmit={...}>
  {({ values, handleChange }) => ...}
</Formik>

// react-hook-form
const { register, handleSubmit } = useForm();
```

## 📚 Learning Resources

- [Expo Documentation](https://docs.expo.dev/)
- [React Native Documentation](https://reactnative.dev/)
- [React Query Docs](https://tanstack.com/query/latest)
- [Zustand Docs](https://github.com/pmndrs/zustand)
- [Zod Docs](https://zod.dev/)

## ✅ Checklist

When choosing a library, check:
- [ ] Active maintenance (recent commits)
- [ ] TypeScript support
- [ ] Expo compatibility
- [ ] Bundle size impact
- [ ] Documentation quality
- [ ] Community size
- [ ] License (MIT preferred)
- [ ] Performance benchmarks
- [ ] Test coverage

---

**Last Updated**: 2024
**Expo SDK**: 51


