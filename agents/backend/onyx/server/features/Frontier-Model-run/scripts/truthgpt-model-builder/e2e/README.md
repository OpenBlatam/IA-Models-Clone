# E2E Tests - TruthGPT Model Builder

## Overview

End-to-end tests for the TruthGPT Model Builder using Playwright.

## Test Structure

- `basic-flow.spec.ts` - Basic user flows and model creation
- `advanced-features.spec.ts` - Advanced features and panels
- `complete-flow.spec.ts` - Complete integration flows
- `accessibility.spec.ts` - Accessibility and screen reader support
- `fixtures/test-data.ts` - Test data and helper functions

## Running Tests

### Run all E2E tests
```bash
npm run test:e2e
```

### Run with UI mode
```bash
npm run test:e2e:ui
```

### Run in headed mode (see browser)
```bash
npm run test:e2e:headed
```

### Debug tests
```bash
npm run test:e2e:debug
```

### Run specific test file
```bash
npx playwright test e2e/basic-flow.spec.ts
```

### Run in specific browser
```bash
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit
```

## Test Coverage

### Basic Flows
- ✅ Model creation from chat
- ✅ Model display in completed list
- ✅ Validation errors
- ✅ Proactive builder activation
- ✅ Queue management
- ✅ Start/pause functionality

### Advanced Features
- ✅ Templates panel
- ✅ Statistics panel
- ✅ Real-time metrics
- ✅ Alerts panel
- ✅ Favorites panel
- ✅ Smart history
- ✅ Quick commands
- ✅ Help panel
- ✅ Keyboard shortcuts

### Complete Integration
- ✅ Full model lifecycle
- ✅ Batch operations
- ✅ Search and filter
- ✅ Export functionality
- ✅ Responsive design

### Accessibility
- ✅ Heading structure
- ✅ Accessible buttons
- ✅ Form inputs
- ✅ Keyboard navigation
- ✅ Color contrast
- ✅ ARIA roles
- ✅ Screen reader support

## Configuration

Tests are configured in `playwright.config.ts`:
- Base URL: `http://localhost:3000`
- Browsers: Chromium, Firefox, WebKit
- Mobile: Chrome, Safari
- Retries: 2 on CI
- Screenshots: On failure
- Videos: On failure

## CI/CD Integration

Tests can be run in CI/CD pipelines:
```bash
npm run test:e2e
```

## Debugging

1. Use `--debug` flag to step through tests
2. Use `--headed` to see browser
3. Use `--ui` for interactive mode
4. Check screenshots and videos in `test-results/`










