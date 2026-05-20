/**
 * Advanced Test Utilities
 * Extended utilities for complex testing scenarios
 */

import { render, RenderOptions, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactElement, ReactNode } from 'react';
import { useMusicStore } from '@/lib/store/music-store';
import { type Track } from '@/lib/api/types';

/**
 * Creates a QueryClient with custom options for testing
 */
export function createTestQueryClient(options?: {
  retry?: boolean;
  cacheTime?: number;
  staleTime?: number;
}) {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: options?.retry ?? false,
        cacheTime: options?.cacheTime ?? 0,
        staleTime: options?.staleTime ?? 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

/**
 * Creates a wrapper component with QueryClient provider
 */
export function createQueryWrapper(options?: Parameters<typeof createTestQueryClient>[0]) {
  const queryClient = createTestQueryClient(options);
  
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}

/**
 * Renders component with all providers
 */
export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'> & {
    queryClientOptions?: Parameters<typeof createTestQueryClient>[0];
  }
) {
  const Wrapper = createQueryWrapper(options?.queryClientOptions);
  
  return render(ui, { wrapper: Wrapper, ...options });
}

/**
 * Waits for store state to match condition
 */
export async function waitForStoreState<T>(
  selector: (state: ReturnType<typeof useMusicStore.getState>) => T,
  condition: (value: T) => boolean,
  options?: { timeout?: number }
) {
  return waitFor(
    () => {
      const state = useMusicStore.getState();
      const value = selector(state);
      expect(condition(value)).toBe(true);
    },
    { timeout: options?.timeout ?? 5000 }
  );
}

/**
 * Creates a mock track for testing
 */
export function createMockTrack(overrides?: Partial<Track>): Track {
  return {
    id: '1',
    name: 'Test Track',
    artists: ['Test Artist'],
    duration_ms: 200000,
    preview_url: 'https://example.com/preview.mp3',
    images: [],
    ...overrides,
  };
}

/**
 * Creates multiple mock tracks
 */
export function createMockTracks(count: number): Track[] {
  return Array.from({ length: count }, (_, i) =>
    createMockTrack({
      id: `${i + 1}`,
      name: `Test Track ${i + 1}`,
      artists: [`Artist ${i + 1}`],
    })
  );
}

/**
 * Resets the music store to initial state
 */
export function resetMusicStore() {
  useMusicStore.setState({
    currentTrack: null,
    playlistQueue: [],
    isPlaying: false,
    volume: 0.5,
    currentTime: 0,
    duration: 0,
  });
}

/**
 * Sets up music store with test data
 */
export function setupMusicStore(data: {
  currentTrack?: Track | null;
  playlistQueue?: Track[];
  isPlaying?: boolean;
  volume?: number;
}) {
  useMusicStore.setState({
    currentTrack: data.currentTrack ?? null,
    playlistQueue: data.playlistQueue ?? [],
    isPlaying: data.isPlaying ?? false,
    volume: data.volume ?? 0.5,
    currentTime: 0,
    duration: 0,
  });
}

/**
 * Waits for element to appear and be visible
 */
export async function waitForElement(
  testId: string,
  options?: { timeout?: number }
) {
  return waitFor(
    () => {
      const element = screen.getByTestId(testId);
      expect(element).toBeVisible();
      return element;
    },
    { timeout: options?.timeout ?? 5000 }
  );
}

/**
 * Waits for text to appear
 */
export async function waitForText(
  text: string | RegExp,
  options?: { timeout?: number }
) {
  return waitFor(
    () => {
      expect(screen.getByText(text)).toBeInTheDocument();
    },
    { timeout: options?.timeout ?? 5000 }
  );
}

/**
 * Simulates user typing with delay
 */
export async function typeWithDelay(
  element: HTMLElement,
  text: string,
  delay: number = 50
) {
  const user = userEvent.setup({ delay });
  await user.type(element, text);
}

/**
 * Simulates user clicking with delay
 */
export async function clickWithDelay(
  element: HTMLElement,
  delay: number = 100
) {
  const user = userEvent.setup({ delay });
  await user.click(element);
}

/**
 * Waits for async operation to complete
 */
export async function waitForAsync(
  fn: () => Promise<void>,
  options?: { timeout?: number; interval?: number }
) {
  return waitFor(
    async () => {
      await fn();
    },
    {
      timeout: options?.timeout ?? 5000,
      interval: options?.interval ?? 100,
    }
  );
}

/**
 * Mocks console methods and restores them after test
 */
export function withConsoleMock(
  mockFn: (method: 'log' | 'error' | 'warn' | 'info') => jest.Mock
) {
  const originalConsole = { ...console };
  const mocks = {
    log: mockFn('log'),
    error: mockFn('error'),
    warn: mockFn('warn'),
    info: mockFn('info'),
  };
  
  Object.assign(console, mocks);
  
  return {
    mocks,
    restore: () => {
      Object.assign(console, originalConsole);
    },
  };
}

/**
 * Creates a delay utility for testing
 */
export function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Waits for condition to be true
 */
export async function waitForCondition(
  condition: () => boolean,
  options?: { timeout?: number; interval?: number }
) {
  return waitFor(
    () => {
      expect(condition()).toBe(true);
    },
    {
      timeout: options?.timeout ?? 5000,
      interval: options?.interval ?? 100,
    }
  );
}

/**
 * Creates a test user event setup with custom options
 */
export function createTestUser(options?: {
  delay?: number;
  skipHover?: boolean;
}) {
  return userEvent.setup({
    delay: options?.delay ?? 0,
    skipHover: options?.skipHover ?? false,
  });
}

/**
 * Asserts that element is accessible
 */
export function expectAccessible(element: HTMLElement) {
  expect(element).toBeInTheDocument();
  expect(element).toBeVisible();
  
  // Check for ARIA attributes if interactive
  if (element.tagName === 'BUTTON' || element.getAttribute('role') === 'button') {
    expect(element).toHaveAttribute('aria-label');
  }
}

/**
 * Creates a mock API response
 */
export function createMockApiResponse<T>(data: T, delay: number = 0) {
  return new Promise<T>((resolve) => {
    setTimeout(() => resolve(data), delay);
  });
}

/**
 * Creates a mock API error
 */
export function createMockApiError(message: string, status: number = 500) {
  const error = new Error(message) as Error & { status?: number };
  error.status = status;
  return Promise.reject(error);
}

/**
 * Waits for API call to complete
 */
export async function waitForApiCall(
  mockFn: jest.Mock,
  options?: { timeout?: number }
) {
  return waitFor(
    () => {
      expect(mockFn).toHaveBeenCalled();
    },
    { timeout: options?.timeout ?? 5000 }
  );
}

/**
 * Asserts that component renders without errors
 */
export function expectNoErrors() {
  expect(console.error).not.toHaveBeenCalled();
}

/**
 * Creates a test environment setup
 */
export function createTestEnvironment() {
  const queryClient = createTestQueryClient();
  resetMusicStore();
  
  return {
    queryClient,
    Wrapper: createQueryWrapper(),
    reset: () => {
      resetMusicStore();
      queryClient.clear();
    },
  };
}

