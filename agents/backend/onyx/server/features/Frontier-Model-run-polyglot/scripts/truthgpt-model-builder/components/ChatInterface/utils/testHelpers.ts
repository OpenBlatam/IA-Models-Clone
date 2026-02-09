/**
 * Test helpers and utilities
 */

import { render, RenderOptions } from '@testing-library/react'
import { ReactElement } from 'react'
import { ChatProvider, SettingsProvider, ThemeProvider } from '../contexts'

/**
 * Custom render with providers
 */
export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <ThemeProvider>
        <SettingsProvider>
          <ChatProvider>
            {children}
          </ChatProvider>
        </SettingsProvider>
      </ThemeProvider>
    )
  }

  return render(ui, { wrapper: Wrapper, ...options })
}

/**
 * Mock localStorage
 */
export function mockLocalStorage() {
  const store: Record<string, string> = {}

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString()
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      Object.keys(store).forEach(key => delete store[key])
    },
    get length() {
      return Object.keys(store).length
    },
    key: (index: number) => {
      const keys = Object.keys(store)
      return keys[index] || null
    },
  }
}

/**
 * Mock sessionStorage
 */
export function mockSessionStorage() {
  return mockLocalStorage()
}

/**
 * Create mock message
 */
export function createMockMessage(overrides?: Partial<any>): any {
  return {
    id: `msg-${Date.now()}-${Math.random()}`,
    role: 'user' as const,
    content: 'Test message',
    timestamp: Date.now(),
    ...overrides,
  }
}

/**
 * Create mock messages array
 */
export function createMockMessages(count: number): any[] {
  return Array.from({ length: count }, (_, i) =>
    createMockMessage({
      id: `msg-${i}`,
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `Message ${i}`,
      timestamp: Date.now() - (count - i) * 1000,
    })
  )
}

/**
 * Wait for async updates
 */
export function waitForAsync(ms: number = 0): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Mock fetch response
 */
export function mockFetchResponse(data: any, status: number = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
    headers: new Headers(),
  } as Response)
}

/**
 * Mock fetch error
 */
export function mockFetchError(message: string = 'Network error') {
  return Promise.reject(new Error(message))
}

/**
 * Create test user event
 */
export function createTestUserEvent() {
  return {
    click: jest.fn(),
    type: jest.fn(),
    clear: jest.fn(),
    selectOptions: jest.fn(),
    deselectOptions: jest.fn(),
    upload: jest.fn(),
    tab: jest.fn(),
    hover: jest.fn(),
    unhover: jest.fn(),
    paste: jest.fn(),
    keyboard: jest.fn(),
  }
}

/**
 * Mock IntersectionObserver
 */
export function mockIntersectionObserver() {
  const mockIntersectionObserver = jest.fn()
  mockIntersectionObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null,
  })
  window.IntersectionObserver = mockIntersectionObserver as any
  return mockIntersectionObserver
}

/**
 * Mock ResizeObserver
 */
export function mockResizeObserver() {
  const mockResizeObserver = jest.fn()
  mockResizeObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null,
  })
  window.ResizeObserver = mockResizeObserver as any
  return mockResizeObserver
}

/**
 * Mock MediaRecorder
 */
export function mockMediaRecorder() {
  const mockMediaRecorder = {
    start: jest.fn(),
    stop: jest.fn(),
    pause: jest.fn(),
    resume: jest.fn(),
    state: 'inactive',
    ondataavailable: null,
    onstop: null,
    onerror: null,
  }

  global.MediaRecorder = jest.fn(() => mockMediaRecorder) as any
  return mockMediaRecorder
}

/**
 * Mock SpeechRecognition
 */
export function mockSpeechRecognition() {
  const mockRecognition = {
    start: jest.fn(),
    stop: jest.fn(),
    abort: jest.fn(),
    continuous: false,
    interimResults: false,
    lang: 'es-ES',
    onresult: null,
    onerror: null,
    onend: null,
  }

  ;(window as any).webkitSpeechRecognition = jest.fn(() => mockRecognition)
  ;(window as any).SpeechRecognition = jest.fn(() => mockRecognition)
  return mockRecognition
}

/**
 * Setup test environment
 */
export function setupTestEnvironment() {
  // Mock localStorage
  const localStorage = mockLocalStorage()
  Object.defineProperty(window, 'localStorage', {
    value: localStorage,
    writable: true,
  })

  // Mock sessionStorage
  const sessionStorage = mockSessionStorage()
  Object.defineProperty(window, 'sessionStorage', {
    value: sessionStorage,
    writable: true,
  })

  // Mock fetch
  global.fetch = jest.fn()

  // Mock IntersectionObserver
  mockIntersectionObserver()

  // Mock ResizeObserver
  mockResizeObserver()

  return {
    localStorage,
    sessionStorage,
    cleanup: () => {
      localStorage.clear()
      sessionStorage.clear()
      jest.clearAllMocks()
    },
  }
}




