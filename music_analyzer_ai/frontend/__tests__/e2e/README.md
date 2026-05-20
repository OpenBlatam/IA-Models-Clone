# E2E Tests - End-to-End Testing Suite

## Overview

This directory contains comprehensive End-to-End (E2E) tests that verify complete user workflows, accessibility, and performance characteristics of the music analyzer application.

## Test Files

### 1. `user-flows.test.tsx`
Tests complete user workflows from start to finish:
- ✅ Search and Play Track Flow
- ✅ Track Analysis Workflow
- ✅ Navigation and API Status Flow
- ✅ Audio Player Controls Flow
- ✅ Error Handling Flow
- ✅ Complete Music Discovery Flow
- ✅ Theme Toggle Flow
- ✅ Multiple Component Interaction Flow

### 2. `music-workflow.test.tsx`
Tests complete music analysis and playback workflows:
- ✅ Complete Track Analysis Workflow
- ✅ Playback Workflow
- ✅ Progress Tracking Workflow
- ✅ Error Recovery Workflow
- ✅ Multi-Step User Journey

### 3. `accessibility.test.tsx`
Tests accessibility features and keyboard navigation:
- ✅ Keyboard Navigation
- ✅ ARIA Attributes
- ✅ Screen Reader Support
- ✅ Focus Management
- ✅ Color Contrast
- ✅ Form Accessibility

### 4. `performance.test.tsx`
Tests performance characteristics and optimizations:
- ✅ Debounce Performance
- ✅ Query Caching
- ✅ Lazy Loading
- ✅ Render Performance
- ✅ Memory Management

## Running E2E Tests

```bash
# Run all E2E tests
npm test -- e2e

# Run specific E2E test file
npm test -- user-flows.test.tsx
npm test -- music-workflow.test.tsx
npm test -- accessibility.test.tsx
npm test -- performance.test.tsx

# Run with coverage
npm run test:coverage -- e2e
```

## Test Structure

Each E2E test file follows this structure:

1. **Setup**: Mock dependencies and create test wrappers
2. **Test Cases**: Individual user flows or scenarios
3. **Assertions**: Verify expected behavior and outcomes
4. **Cleanup**: Reset mocks and timers

## Best Practices

1. **Real User Scenarios**: Tests simulate actual user behavior
2. **Complete Workflows**: Tests cover entire user journeys
3. **Error Scenarios**: Tests include error handling and recovery
4. **Accessibility**: Tests verify a11y compliance
5. **Performance**: Tests verify optimization behaviors

## Coverage

E2E tests cover:
- ✅ Complete user workflows
- ✅ Component interactions
- ✅ API integrations
- ✅ Error handling
- ✅ Accessibility features
- ✅ Performance optimizations
- ✅ Keyboard navigation
- ✅ Screen reader support

## Notes

- E2E tests use `jest.useFakeTimers()` for time-based operations
- Tests mock external API calls for reliability
- Tests verify both happy paths and error scenarios
- Accessibility tests ensure WCAG compliance
- Performance tests verify optimization strategies

