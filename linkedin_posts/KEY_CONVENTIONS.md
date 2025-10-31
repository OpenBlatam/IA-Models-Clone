# Key Conventions for LinkedIn Posts React Native/Expo Application

## Code Style and Structure

### General Principles
- Write concise, technical TypeScript code with accurate examples
- Use functional and declarative programming patterns; avoid classes
- Prefer iteration and modularization over code duplication
- Use descriptive variable names with auxiliary verbs (e.g., `isLoading`, `hasError`)
- Structure files: exported component, subcomponents, helpers, static content, types
- Avoid global variables to prevent unintended side effects
- Use ES6+ features like arrow functions, destructuring, and template literals
- Use PropTypes for type checking if not using TypeScript

### Component Organization
- Break down components into smaller, reusable pieces
- Keep components focused on a single responsibility
- Organize files by feature (e.g., `user-profile`, `chat-screen`)
- Use the "function" keyword for pure functions
- Avoid unnecessary curly braces in conditionals; use concise syntax for simple statements
- Use declarative JSX
- Use Prettier for consistent code formatting

## Naming Conventions

### Variables and Functions
- Use camelCase for variables and functions (e.g., `isFetchingData`, `handleUserInput`)
- Use PascalCase for component names (e.g., `UserProfile`, `ChatScreen`)
- Use lowercase with dashes for directories (e.g., `components/auth-wizard`)
- Favor named exports for components

## TypeScript Usage

### Type Safety
- Use TypeScript for all code; prefer interfaces over types
- Avoid enums; use maps instead
- Use functional components with TypeScript interfaces
- Use strict mode in TypeScript for better type safety

## Performance Optimization

### State Management
- Minimize the use of `useState` and `useEffect`; prefer context and reducers for state management
- Optimize state management: avoid unnecessary state updates and use local state only when needed
- Use React.memo() for functional components to prevent unnecessary re-renders

### Rendering Optimization
- Optimize FlatList with props like `removeClippedSubviews`, `maxToRenderPerBatch`, and `windowSize`
- Avoid anonymous functions in `renderItem` or event handlers to prevent re-renders
- Avoid unnecessary re-renders by memoizing components and using `useMemo` and `useCallback` hooks appropriately

### App Performance
- Use Expo's AppLoading and SplashScreen for optimized app startup experience
- Optimize images: use WebP format where supported, include size data, implement lazy loading with expo-image
- Implement code splitting and lazy loading for non-critical components with React's Suspense and dynamic imports
- Profile and monitor performance using React Native's built-in tools and Expo's debugging features

## UI and Styling

### Component Libraries
- Use Expo's built-in components for common UI patterns and layouts
- Implement responsive design with Flexbox and Expo's `useWindowDimensions` for screen size adjustments
- Use styled-components or Tailwind CSS for component styling
- Implement dark mode support using Expo's `useColorScheme`
- Ensure high accessibility (a11y) standards using ARIA roles and native accessibility props
- Leverage react-native-reanimated and react-native-gesture-handler for performant animations and gestures

### Consistent Styling
- Use StyleSheet.create() for consistent styling or Styled Components for dynamic styles
- Responsive design: ensure your design adapts to various screen sizes and orientations
- Consider using responsive units and libraries like react-native-responsive-screen
- Optimize image handling: use optimized image libraries like react-native-fast-image to handle images efficiently

## Safe Area Management

### Global Safe Area Setup
- Use SafeAreaProvider from react-native-safe-area-context to manage safe areas globally in your app
- Wrap top-level components with SafeAreaView to handle notches, status bars, and other screen insets on both iOS and Android
- Use SafeAreaScrollView for scrollable content to ensure it respects safe area boundaries
- Avoid hardcoding padding or margins for safe areas; rely on SafeAreaView and context hooks

## Navigation

### Routing and Navigation
- Use react-navigation for routing and navigation; follow its best practices for stack, tab, and drawer navigators
- Leverage deep linking and universal links for better user engagement and navigation flow
- Use dynamic routes with expo-router for better navigation handling

## State Management

### Global State
- Use React Context and useReducer for managing global state
- Leverage react-query for data fetching and caching; avoid excessive API calls
- For complex state management, consider using Zustand or Redux Toolkit
- Handle URL search parameters using libraries like expo-linking

## Error Handling and Validation

### Runtime Validation
- Use Zod for runtime validation and error handling
- Implement proper error logging using Sentry or a similar service

### Error Handling Patterns
- Prioritize error handling and edge cases:
  - Handle errors at the beginning of functions
  - Use early returns for error conditions to avoid deeply nested if statements
  - Avoid unnecessary else statements; use if-return pattern instead
  - Implement global error boundaries to catch and handle unexpected errors
- Use expo-error-reporter for logging and reporting errors in production

## Testing

### Testing Strategy
- Write unit tests using Jest and React Native Testing Library
- Implement integration tests for critical user flows using Detox
- Use Expo's testing tools for running tests in different environments
- Consider snapshot testing for components to ensure UI consistency

## Security

### Input Security
- Sanitize user inputs to prevent XSS attacks
- Use react-native-encrypted-storage for secure storage of sensitive data
- Ensure secure communication with APIs using HTTPS and proper authentication
- Use Expo's Security guidelines to protect your app: https://docs.expo.dev/guides/security/

## Internationalization (i18n)

### Localization Support
- Use react-native-i18n or expo-localization for internationalization and localization
- Support multiple languages and RTL layouts
- Ensure text scaling and font adjustments for accessibility

## Expo Best Practices

### Managed Workflow
1. Rely on Expo's managed workflow for streamlined development and deployment
2. Prioritize Mobile Web Vitals (Load Time, Jank, and Responsiveness)
3. Use expo-constants for managing environment variables and configuration
4. Use expo-permissions to handle device permissions gracefully
5. Implement expo-updates for over-the-air (OTA) updates
6. Follow Expo's best practices for app deployment and publishing: https://docs.expo.dev/distribution/introduction/
7. Ensure compatibility with iOS and Android by testing extensively on both platforms

### Documentation and Resources
- Use Expo's official documentation for setting up and configuring your projects: https://docs.expo.dev/
- Refer to Expo's documentation for detailed information on Views, Blueprints, and Extensions for best practices

## Accessibility Features

### Text Scaling and Font Adjustments
- Implement comprehensive text scaling and font adjustments for accessibility
- Use system font scale preferences
- Support bold text preferences
- Implement responsive font sizing based on device characteristics
- Ensure minimum and maximum font size limits
- Support screen reader compatibility (VoiceOver/TalkBack)
- Implement high contrast mode support
- Use WCAG-compliant contrast ratios
- Ensure minimum touch target sizes (44pt for iOS, 48pt for Android)

### Accessibility Components
- AccessibleText: Enhanced Text component with built-in accessibility features
- AccessibleButton: Touchable button with proper accessibility labels and states
- AccessibleView: Container component with accessibility support
- AccessibilityManager: Singleton class for managing accessibility settings
- useAccessibility: React hook for accessing accessibility preferences
- useScaledFontSize: Hook for reactive font size scaling
- useResponsiveAccessibility: Hook for responsive accessibility features

### Accessibility Utilities
- AccessibilityUtils: WCAG-compliant contrast checking and color accessibility
- accessibilityStyles: Pre-defined styles for touch targets and typography
- Responsive accessibility with tablet and landscape support
- Platform-specific optimizations for iOS and Android

## API Documentation

### Core Entities
```typescript
interface LinkedInPost {
  readonly id: string;
  readonly userId: string;
  readonly title: string;
  readonly content: PostContent;
  readonly type: PostType;
  readonly tone: PostTone;
  readonly status: PostStatus;
  readonly engagementMetrics: EngagementMetrics;
  readonly createdAt: Date;
  readonly updatedAt: Date;
}

interface PostContent {
  readonly text: string;
  readonly hashtags: string[];
  readonly mentions: string[];
  readonly mediaUrls?: string[];
}

interface EngagementMetrics {
  readonly views: number;
  readonly likes: number;
  readonly comments: number;
  readonly shares: number;
  readonly clicks: number;
}
```

### Service Layer
```typescript
class PostService {
  async createPost(request: PostGenerationRequest): Promise<LinkedInPost>
  async updatePost(postId: string, updates: Partial<LinkedInPost>): Promise<LinkedInPost>
  async getPost(postId: string): Promise<LinkedInPost | null>
  async listPosts(userId: string, filters?: PostFilters): Promise<LinkedInPost[]>
  async deletePost(postId: string): Promise<boolean>
  async optimizePost(postId: string): Promise<PostOptimizationResult>
  async validatePostRequest(request: PostGenerationRequest): Promise<PostValidationResult>
  async generatePostAnalytics(postId: string): Promise<ContentAnalysisResult>
}
```

### State Management
```typescript
interface PostState {
  posts: LinkedInPost[];
  currentPost: LinkedInPost | null;
  isLoading: boolean;
  error: string | null;
  filters: PostFilters;
}

const usePostStore = create<PostState>()(
  immer((set, get) => ({
    // State implementation
  }))
);
```

This comprehensive set of conventions ensures a professional, maintainable, and accessible React Native/Expo application following industry best practices. 