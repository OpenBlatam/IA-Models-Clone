# Hooks Guide

Complete guide to all custom hooks available in the application.

## 📚 Table of Contents

1. [Core Hooks](#core-hooks)
2. [Utility Hooks](#utility-hooks)
3. [Storage Hooks](#storage-hooks)
4. [Network & Status Hooks](#network--status-hooks)
5. [UI Hooks](#ui-hooks)
6. [Timer Hooks](#timer-hooks)
7. [Platform Hooks](#platform-hooks)

## Core Hooks

### `useForm<T>`
Form handling with Zod validation.

```typescript
const form = useForm({
  initialValues: { email: '', password: '' },
  validationSchema: loginSchema,
  onSubmit: async (values) => {
    await login(values.email, values.password);
  },
});

// Usage
<Input
  value={form.values.email}
  onChangeText={form.handleChange('email')}
  onBlur={form.handleBlur('email')}
  error={form.touched.email ? form.errors.email : undefined}
/>
<Button onPress={form.handleSubmit} disabled={!form.isValid} />
```

### `useAuthenticatedQuery<T>`
React Query hook that automatically includes user authentication.

```typescript
const { data, isLoading } = useAuthenticatedQuery({
  queryKey: ['dashboard'],
  queryFn: (userId) => apiService.getDashboard(userId),
});
```

### `useAuthenticatedMutation<T>`
React Query mutation hook with authentication.

```typescript
const mutation = useAuthenticatedMutation({
  mutationFn: (userId, variables) => apiService.updateProfile(userId, variables),
});
```

### `useResponsive`
Get responsive breakpoints and screen info.

```typescript
const { isSmall, isMedium, isLarge, isTablet, isPhone, width, height } = useResponsive();
```

## Utility Hooks

### `useDebounce<T>`
Debounce a value to reduce frequent updates.

```typescript
const [searchTerm, setSearchTerm] = useState('');
const debouncedSearchTerm = useDebounce(searchTerm, 500);

useEffect(() => {
  // Search API call only after 500ms of no typing
  searchJobs(debouncedSearchTerm);
}, [debouncedSearchTerm]);
```

### `useThrottle<T>`
Throttle a function to limit execution frequency.

```typescript
const handleScroll = useThrottle((event) => {
  // Called at most once every 300ms
  updateScrollPosition(event);
}, 300);
```

### `usePrevious<T>`
Get the previous value of a state or prop.

```typescript
const [count, setCount] = useState(0);
const prevCount = usePrevious(count);

// prevCount will be undefined on first render, then previous value
```

### `useToggle`
Toggle boolean state with helper methods.

```typescript
const [isOpen, { toggle, setTrue, setFalse }] = useToggle(false);

// Usage
<Button onPress={toggle} title={isOpen ? 'Close' : 'Open'} />
```

### `useAsync<T>`
Handle async operations with loading and error states.

```typescript
const { data, error, isLoading, execute, reset } = useAsync(
  async (id: string) => {
    return await fetchUser(id);
  },
  false // Don't execute immediately
);

// Execute manually
execute('user123');

// Reset state
reset();
```

## Storage Hooks

### `useSecureStorage<T>`
Secure encrypted storage with React state.

```typescript
const { value, setValue, removeValue, isLoading, error } = useSecureStorage<string>(
  'user_token',
  null
);

// Usage
setValue('new_token');
removeValue();
```

## Network & Status Hooks

### `useNetworkStatus`
Monitor network connectivity.

```typescript
const { isConnected, type, isInternetReachable } = useNetworkStatus();

if (!isConnected) {
  return <OfflineMessage />;
}
```

### `useAppState`
Monitor app state (active, background, inactive).

```typescript
const { appState, isActive, isBackground } = useAppState();

useEffect(() => {
  if (isActive) {
    // App came to foreground
    refreshData();
  }
}, [isActive]);
```

## UI Hooks

### `useKeyboard`
Monitor keyboard visibility and height.

```typescript
const { isVisible, height } = useKeyboard();

return (
  <View style={{ paddingBottom: isVisible ? height : 0 }}>
    {/* Content */}
  </View>
);
```

### `useFocus`
Execute callback when screen is focused.

```typescript
useFocus(() => {
  // Refresh data when screen is focused
  refetch();
}, [refetch]);
```

### `useFocusRef`
Get a ref that tracks if screen is focused.

```typescript
const isFocusedRef = useFocusRef();

// Check if focused
if (isFocusedRef.current) {
  // Do something
}
```

### `useAnimation`
Create and control animations.

```typescript
const { value, animate, spring, reset } = useAnimation(0);

// Animate
animate(100, 300).start();

// Spring animation
spring(100, { tension: 50, friction: 7 }).start();

// Use in component
<Animated.View style={{ opacity: value }} />
```

### `useOrientation`
Monitor device orientation.

```typescript
const { orientation, isPortrait, isLandscape, dimensions } = useOrientation();

if (isLandscape) {
  return <LandscapeLayout />;
}
```

### `useDimensions`
Get screen dimensions with updates.

```typescript
const { width, height, scale, fontScale } = useDimensions();
```

### `useSafeArea`
Get safe area insets.

```typescript
const { top, bottom, left, right } = useSafeArea();

return (
  <View style={{ paddingTop: top, paddingBottom: bottom }}>
    {/* Content */}
  </View>
);
```

## Timer Hooks

### `useInterval`
Execute callback at intervals.

```typescript
const [count, setCount] = useState(0);

useInterval(() => {
  setCount(c => c + 1);
}, 1000); // Every second

// Stop interval by passing null
useInterval(() => {}, null);
```

### `useTimeout`
Execute callback after delay.

```typescript
useTimeout(() => {
  // Execute after 3 seconds
  navigateToNextScreen();
}, 3000);

// Cancel by passing null
useTimeout(() => {}, null);
```

## Platform Hooks

### `usePermissions`
Request and check permissions.

```typescript
const {
  granted,
  canAskAgain,
  status,
  isLoading,
  requestPermission,
  checkPermission,
} = usePermissions(Permissions.CAMERA);

if (!granted) {
  return <Button onPress={requestPermission} title="Grant Camera Permission" />;
}
```

### `useImagePicker`
Pick images from camera or library.

```typescript
const { image, pickImage, clearImage, isLoading, error } = useImagePicker({
  allowsEditing: true,
  aspect: [4, 3],
  quality: 0.8,
});

// Pick from library
pickImage('library');

// Pick from camera
pickImage('camera');

// Display image
{image && <Image source={{ uri: image.uri }} />}
```

### `useDeepLink`
Handle deep linking and navigation.

```typescript
const { initialUrl, currentUrl } = useDeepLink();

// Automatically handles routing based on URL
// myapp://dashboard -> navigates to dashboard
// myapp://jobs/123 -> navigates to job details
```

### `useBackHandler`
Handle Android back button.

```typescript
useBackHandler(() => {
  // Return true to prevent default back behavior
  if (shouldPreventBack) {
    showConfirmDialog();
    return true;
  }
  return false;
});
```

### `useClipboard`
Copy and read from clipboard.

```typescript
const { clipboardContent, copyToClipboard, getClipboardContent } = useClipboard();

// Copy
await copyToClipboard('Hello World');

// Read
const text = await getClipboardContent();
```

### `useVibration`
Trigger haptic feedback.

```typescript
const { vibrate } = useVibration();

// Different vibration types
vibrate('light');
vibrate('medium');
vibrate('heavy');
vibrate('success');
vibrate('warning');
vibrate('error');
```

### `useLocation`
Get device location.

```typescript
const {
  location,
  isLoading,
  error,
  hasPermission,
  getCurrentLocation,
  requestPermission,
} = useLocation();

// Get location
await getCurrentLocation();

// Use location
{location && (
  <Text>
    {location.latitude}, {location.longitude}
  </Text>
)}
```

### `useBiometrics`
Biometric authentication (Face ID, Touch ID, Fingerprint).

```typescript
const {
  isAvailable,
  biometricType,
  isLoading,
  checkAvailability,
  authenticate,
} = useBiometrics();

// Check if available
useEffect(() => {
  checkAvailability();
}, []);

// Authenticate
const result = await authenticate('Please authenticate to continue');
if (result.success) {
  // Authentication successful
}
```

## Best Practices

1. **Always handle loading states**: Most hooks provide `isLoading` state
2. **Handle errors**: Check `error` states and provide user feedback
3. **Cleanup**: Hooks handle cleanup automatically, but be aware of subscriptions
4. **Performance**: Use `useDebounce` and `useThrottle` for expensive operations
5. **Permissions**: Always request permissions before using platform features
6. **Type safety**: All hooks are fully typed with TypeScript

## Examples

### Search with Debounce
```typescript
function SearchScreen() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 500);
  
  const { data } = useQuery({
    queryKey: ['search', debouncedQuery],
    queryFn: () => searchAPI(debouncedQuery),
    enabled: debouncedQuery.length > 0,
  });
  
  return <Input value={query} onChangeText={setQuery} />;
}
```

### Image Upload
```typescript
function ProfileScreen() {
  const { image, pickImage, isLoading } = useImagePicker();
  const { uploadImage } = useMutation();
  
  const handlePickImage = async () => {
    await pickImage('library');
    if (image) {
      await uploadImage(image.uri);
    }
  };
  
  return (
    <View>
      {image && <Image source={{ uri: image.uri }} />}
      <Button onPress={handlePickImage} title="Pick Image" />
    </View>
  );
}
```

### Network-Aware Component
```typescript
function DataScreen() {
  const { isConnected } = useNetworkStatus();
  const { data } = useQuery({
    queryKey: ['data'],
    queryFn: fetchData,
    enabled: isConnected,
  });
  
  if (!isConnected) {
    return <OfflineMessage />;
  }
  
  return <DataList data={data} />;
}
```

---

**Note**: Make sure to install required dependencies for platform-specific hooks:
- `@react-native-community/netinfo` for `useNetworkStatus`
- `expo-image-picker` for `useImagePicker`
- `expo-location` for `useLocation`
- `expo-local-authentication` for `useBiometrics`
- And others as needed


