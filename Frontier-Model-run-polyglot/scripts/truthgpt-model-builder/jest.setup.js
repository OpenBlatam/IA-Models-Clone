/**
 * Jest Setup
 */

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
})

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
}
global.localStorage = localStorageMock

// Mock Notification
global.Notification = jest.fn().mockImplementation(() => ({
  close: jest.fn(),
}))

// Mock fetch
global.fetch = jest.fn()

// Mock URL.createObjectURL
global.URL.createObjectURL = jest.fn(() => 'blob:mock-url')
global.URL.revokeObjectURL = jest.fn()

// Mock document.createElement for download
const mockClick = jest.fn()
global.document.createElement = jest.fn((tag) => {
  if (tag === 'a') {
    return {
      href: '',
      download: '',
      click: mockClick,
    }
  }
  return {}
})

global.document.body.appendChild = jest.fn()
global.document.body.removeChild = jest.fn()










