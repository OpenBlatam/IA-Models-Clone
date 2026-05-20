export const config = {
  api: {
    baseUrl: process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:8000',
    endpoints: {
      generate: '/api/key-messages/generate',
      analyze: '/api/key-messages/analyze'
    }
  },
  defaults: {
    maxLength: 10000,
    messageType: 'informational',
    tone: 'professional'
  },
  errorMessages: {
    generationError: 'Failed to generate message. Please try again.',
    networkError: 'Network error. Please check your connection.',
    serverError: 'Server error. Please try again later.',
    validationError: 'Invalid input. Please check your message.'
  }
}; 