# Playwright E2E Tests

This directory contains end-to-end tests for the Dermatology AI frontend application using Playwright.

## Setup

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Install Playwright browsers:
```bash
npx playwright install
```

## Running Tests

### Run all tests
```bash
npm run test
```

### Run tests in UI mode
```bash
npm run test:ui
```

### Run tests in debug mode
```bash
npm run test:debug
```

### Run tests in headed mode (see browser)
```bash
npm run test:headed
```

### Run tests for specific browser
```bash
npm run test:chromium
npm run test:firefox
npm run test:webkit
```

## Test Structure

- `home.spec.ts` - Tests for the home page including image upload and analysis
- `auth.spec.ts` - Tests for authentication (login and register)
- `dashboard.spec.ts` - Tests for the dashboard page
- `navigation.spec.ts` - Tests for navigation and header functionality
- `helpers/test-helpers.ts` - Reusable helper functions and constants
- `fixtures/auth.fixture.ts` - Custom fixtures for authenticated tests

## Test Best Practices

- Tests use role-based locators (`getByRole`, `getByLabel`, `getByText`) instead of CSS selectors
- Tests are isolated and can run in parallel
- Tests use web-first assertions (`toBeVisible`, `toHaveText`, etc.)
- Tests include proper error handling and clear failure messages
- Tests follow real user behavior patterns

## Configuration

The Playwright configuration is in `playwright.config.ts` at the root of the project. It includes:
- Multiple browser projects (Chromium, Firefox, WebKit, Mobile Chrome, Mobile Safari)
- Automatic server startup for local development
- Screenshot and video capture on failures
- Trace collection for debugging

## Environment Variables

- `PLAYWRIGHT_TEST_BASE_URL` - Override the base URL for tests (default: http://localhost:3000)
- `CI` - Set to true in CI environments to enable retries and other CI-specific settings



