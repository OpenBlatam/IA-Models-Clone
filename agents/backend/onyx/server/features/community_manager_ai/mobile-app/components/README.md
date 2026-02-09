# Components Documentation

## Component Structure

All components follow a consistent structure:
- Named exports
- TypeScript interfaces
- Proper prop types
- Accessibility support
- Performance optimized

## Component Categories

### UI Components (`components/ui/`)
Basic reusable UI components:
- `AccessibleButton` - Button with full accessibility
- `AccessibleText` - Text with semantic variants
- `Input` - Text input with validation
- `TextArea` - Multi-line input
- `Checkbox` - Checkbox component
- `Button` - Standard button
- `Card` - Card container
- `Loading` - Loading indicator
- `EmptyState` - Empty state display

### Optimized Components (`components/optimized/`)
Performance-optimized components:
- `MemoizedList` - Optimized FlatList
- `LazyImage` - Lazy-loaded images

### Animation Components (`components/animations/`)
Animated components:
- `FadeInView` - Fade in animation
- `AnimatedView` - General animation wrapper

### Gesture Components (`components/gestures/`)
Gesture-enabled components:
- `SwipeableCard` - Swipeable card with gestures

### Lazy Components (`components/lazy/`)
Code-splitting components:
- `SuspenseWrapper` - Suspense boundary wrapper

## Usage Examples

### AccessibleButton
```typescript
<AccessibleButton
  title="Submit"
  onPress={handleSubmit}
  accessibilityLabel="Submit form"
  accessibilityHint="Submits the current form"
/>
```

### MemoizedList
```typescript
<MemoizedList
  data={items}
  renderItem={({ item }) => <ItemComponent item={item} />}
  keyExtractor={(item) => item.id}
/>
```

### FadeInView
```typescript
<FadeInView delay={100} from="bottom">
  <Text>Animated content</Text>
</FadeInView>
```

### SwipeableCard
```typescript
<SwipeableCard
  onSwipeLeft={() => handleDelete()}
  onSwipeRight={() => handleEdit()}
>
  <CardContent />
</SwipeableCard>
```

## Best Practices

1. Always use TypeScript interfaces for props
2. Include accessibility props
3. Memoize expensive components
4. Use lazy loading for heavy components
5. Follow the component structure pattern


