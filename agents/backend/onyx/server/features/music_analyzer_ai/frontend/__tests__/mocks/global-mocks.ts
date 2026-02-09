/**
 * Global Mocks
 * Centralized mocks for consistent testing
 */

// Mock next/navigation
export const mockRouter = {
  push: jest.fn(),
  replace: jest.fn(),
  prefetch: jest.fn(),
  back: jest.fn(),
  pathname: '/music',
  query: {},
};

// Mock react-hot-toast
export const mockToast = {
  success: jest.fn(),
  error: jest.fn(),
  info: jest.fn(),
  loading: jest.fn(),
};

// Mock framer-motion
export const mockFramerMotion = {
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    span: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  },
  AnimatePresence: ({ children }: any) => children,
  useAnimation: () => ({
    start: jest.fn(),
    stop: jest.fn(),
  }),
};

// Mock window.matchMedia
export const mockMatchMedia = (matches = false) => {
  return jest.fn().mockImplementation((query) => ({
    matches,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  }));
};

// Mock IntersectionObserver
export const mockIntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
};

// Mock Audio API
export const mockAudio = () => {
  return jest.fn().mockImplementation(() => ({
    play: jest.fn().mockResolvedValue(undefined),
    pause: jest.fn(),
    load: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    currentTime: 0,
    duration: 100,
    volume: 1,
    muted: false,
    paused: true,
  }));
};

// Mock localStorage
export const mockLocalStorage = () => {
  const store: Record<string, string> = {};

  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value.toString();
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      Object.keys(store).forEach((key) => delete store[key]);
    }),
  };
};

// Setup global mocks
export const setupGlobalMocks = () => {
  // Setup matchMedia
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: mockMatchMedia(false),
  });

  // Setup IntersectionObserver
  global.IntersectionObserver = mockIntersectionObserver;

  // Setup Audio
  global.Audio = mockAudio() as any;

  // Setup localStorage
  const storage = mockLocalStorage();
  Object.defineProperty(window, 'localStorage', {
    value: storage,
  });
};

