# Advanced Integration Testing Guide

## 📋 Overview

This guide covers advanced integration testing patterns for the Music Analyzer AI frontend, including error boundaries, router integration, context integration, and advanced test utilities.

## 🎯 Test Files

### 1. Error Boundary Integration (`error-boundary-integration.test.tsx`)

Tests the integration between Error Boundary components and:
- Regular React components
- Zustand store
- API services
- Nested components
- Async operations

**Key Test Scenarios:**
- Error catching and display
- Error recovery
- Custom fallback components
- Error boundary levels (page vs component)
- Nested error boundaries
- Async error handling

**Example:**
```typescript
it('should catch and display component errors', () => {
  render(
    <ErrorBoundary>
      <ThrowingComponent shouldThrow={true} />
    </ErrorBoundary>
  );
  
  expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
});
```

### 2. Router Integration (`router-integration.test.tsx`)

Tests Next.js router integration with:
- Component navigation
- Store state persistence
- Search params
- Error handling

**Key Test Scenarios:**
- Basic navigation (push, back, refresh)
- Navigation with store state
- Search params reading
- Router error handling
- State persistence across navigation

**Example:**
```typescript
it('should navigate with track in store', async () => {
  const user = userEvent.setup();
  
  render(<RouterStoreComponent />);
  
  await user.click(screen.getByTestId('navigate-with-track'));
  
  expect(mockPush).toHaveBeenCalledWith('/music');
  expect(useMusicStore.getState().currentTrack).toBeTruthy();
});
```

### 3. Context Integration (`context-integration.test.tsx`)

Tests React Context integration with:
- Theme context
- User context
- Store integration
- Multiple contexts

**Key Test Scenarios:**
- Basic context usage
- Context updates
- Context with store
- Multiple contexts
- Context provider nesting
- State persistence

**Example:**
```typescript
it('should work with store and context together', async () => {
  const user = userEvent.setup();
  
  render(<ThemeStoreComponent />);
  
  await user.click(screen.getByTestId('set-track'));
  
  await waitFor(() => {
    expect(screen.getByTestId('track')).toHaveTextContent('Theme Track');
  });
});
```

## 🛠️ Advanced Test Utilities

### `advanced-test-utils.tsx`

Extended utilities for complex testing scenarios:

#### Query Client Utilities
```typescript
// Create test query client with custom options
const queryClient = createTestQueryClient({
  retry: false,
  cacheTime: 0,
  staleTime: 0,
});

// Create wrapper component
const Wrapper = createQueryWrapper();
```

#### Store Utilities
```typescript
// Wait for store state
await waitForStoreState(
  (state) => state.currentTrack,
  (track) => track !== null
);

// Reset store
resetMusicStore();

// Setup store with test data
setupMusicStore({
  currentTrack: mockTrack,
  isPlaying: true,
});
```

#### Mock Data Utilities
```typescript
// Create mock track
const track = createMockTrack({
  id: '1',
  name: 'Custom Track',
});

// Create multiple tracks
const tracks = createMockTracks(5);
```

#### Async Utilities
```typescript
// Wait for async operation
await waitForAsync(async () => {
  await someAsyncFunction();
});

// Wait for condition
await waitForCondition(() => {
  return someCondition();
});
```

#### User Event Utilities
```typescript
// Type with delay
await typeWithDelay(input, 'test query', 50);

// Click with delay
await clickWithDelay(button, 100);

// Create test user
const user = createTestUser({ delay: 100 });
```

#### API Mocking Utilities
```typescript
// Mock API response
const response = createMockApiResponse(data, 100);

// Mock API error
const error = createMockApiError('Not found', 404);
```

#### Console Mocking
```typescript
// Mock console methods
const { mocks, restore } = withConsoleMock((method) => jest.fn());
// ... test code ...
restore();
```

#### Test Environment
```typescript
// Create complete test environment
const env = createTestEnvironment();
render(<Component />, { wrapper: env.Wrapper });
env.reset(); // Clean up
```

## 📚 Best Practices

### 1. Error Boundary Testing
- Always suppress console.error for error boundary tests
- Test both error catching and recovery
- Test nested error boundaries
- Test custom fallback components

### 2. Router Testing
- Mock Next.js router functions
- Test navigation with state
- Test search params handling
- Test error scenarios

### 3. Context Testing
- Test context provider setup
- Test context updates
- Test multiple contexts
- Test context with store

### 4. Test Utilities
- Use advanced utilities for complex scenarios
- Create reusable test data
- Use waitFor utilities for async operations
- Clean up after tests

## 🎨 Test Patterns

### Pattern 1: Error Boundary with Store
```typescript
it('should handle store errors', () => {
  render(
    <ErrorBoundary>
      <StoreComponent shouldThrow={true} />
    </ErrorBoundary>
  );
  
  expect(screen.getByText(/error/i)).toBeInTheDocument();
});
```

### Pattern 2: Router with Store
```typescript
it('should navigate with state', async () => {
  const user = userEvent.setup();
  
  render(<RouterStoreComponent />);
  
  await user.click(screen.getByTestId('navigate'));
  
  expect(mockPush).toHaveBeenCalled();
  expect(useMusicStore.getState().currentTrack).toBeTruthy();
});
```

### Pattern 3: Multiple Contexts
```typescript
it('should work with multiple contexts', () => {
  render(
    <ThemeProvider>
      <UserProvider>
        <MultiContextComponent />
      </UserProvider>
    </ThemeProvider>
  );
  
  expect(screen.getByTestId('theme')).toBeInTheDocument();
  expect(screen.getByTestId('user')).toBeInTheDocument();
});
```

## 🚀 Running Tests

```bash
# Run all integration tests
npm test -- integration

# Run specific test file
npm test -- error-boundary-integration

# Run with coverage
npm test -- integration --coverage
```

## 📊 Coverage Goals

- Error Boundary Integration: 100%
- Router Integration: 100%
- Context Integration: 100%
- Advanced Utilities: 100%

## 🔍 Debugging Tips

1. **Error Boundary Issues:**
   - Check console.error suppression
   - Verify error boundary level
   - Test custom fallback

2. **Router Issues:**
   - Verify router mocks
   - Check navigation calls
   - Test state persistence

3. **Context Issues:**
   - Verify provider setup
   - Check context updates
   - Test multiple contexts

4. **Utility Issues:**
   - Check async operations
   - Verify mock data
   - Test cleanup

## 📝 Examples

See the test files for complete examples:
- `error-boundary-integration.test.tsx`
- `router-integration.test.tsx`
- `context-integration.test.tsx`
- `advanced-test-utils.tsx`

## 🎯 Next Steps

1. Add more integration scenarios
2. Extend advanced utilities
3. Add performance tests
4. Add accessibility tests

---

**Last Updated**: Latest
**Version**: 1.0.0

