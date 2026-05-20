/**
 * Error Boundary Integration Tests
 * Tests for error boundary integration with components, store, and API
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { useMusicStore } from '@/lib/store/music-store';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/types';

// Mock API service
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
    analyzeTrack: jest.fn(),
  },
}));

const mockSearchTracks = musicApiService.searchTracks as jest.MockedFunction<
  typeof musicApiService.searchTracks
>;

// Component that throws error
const ThrowingComponent = ({ shouldThrow = false }: { shouldThrow?: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>No error</div>;
};

// Component that uses store and might throw
const StoreComponent = ({ shouldThrow = false }: { shouldThrow?: boolean }) => {
  const { setCurrentTrack } = useMusicStore();
  
  if (shouldThrow) {
    throw new Error('Store error');
  }
  
  const handleClick = () => {
    const track: Track = {
      id: '1',
      name: 'Test',
      artists: ['Artist'],
      duration_ms: 200000,
      preview_url: 'https://example.com/preview.mp3',
      images: [],
    };
    setCurrentTrack(track);
  };
  
  return <button onClick={handleClick}>Set Track</button>;
};

// Component that uses API and might throw
const ApiComponent = ({ shouldThrow = false }: { shouldThrow?: boolean }) => {
  if (shouldThrow) {
    throw new Error('API error');
  }
  
  const handleSearch = async () => {
    await mockSearchTracks('test');
  };
  
  return <button onClick={handleSearch}>Search</button>;
};

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('Error Boundary Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Suppress console.error for error boundary tests
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Error Boundary with Components', () => {
    it('should catch and display component errors', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary>
            <ThrowingComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    });

    it('should allow recovery after error', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      const { rerender } = render(
        <Wrapper>
          <ErrorBoundary>
            <ThrowingComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
      
      // Reset error boundary
      rerender(
        <Wrapper>
          <ErrorBoundary>
            <ThrowingComponent shouldThrow={false} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      await waitFor(() => {
        expect(screen.getByText('No error')).toBeInTheDocument();
      });
    });

    it('should call onError callback when error occurs', () => {
      const onError = jest.fn();
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary onError={onError}>
            <ThrowingComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(onError).toHaveBeenCalledWith(
        expect.any(Error),
        expect.objectContaining({
          componentStack: expect.any(String),
        })
      );
    });
  });

  describe('Error Boundary with Store', () => {
    it('should catch errors in components using store', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary>
            <StoreComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    });

    it('should allow store operations after error recovery', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      const { rerender } = render(
        <Wrapper>
          <ErrorBoundary>
            <StoreComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      // Recover from error
      rerender(
        <Wrapper>
          <ErrorBoundary>
            <StoreComponent shouldThrow={false} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      const button = await screen.findByText('Set Track');
      await user.click(button);
      
      const { currentTrack } = useMusicStore.getState();
      expect(currentTrack).toBeTruthy();
      expect(currentTrack?.name).toBe('Test');
    });
  });

  describe('Error Boundary with API', () => {
    it('should catch errors in components using API', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary>
            <ApiComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    });

    it('should handle API errors gracefully', async () => {
      const user = userEvent.setup();
      mockSearchTracks.mockRejectedValueOnce(new Error('API Error'));
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary>
            <ApiComponent shouldThrow={false} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      const button = screen.getByText('Search');
      await user.click(button);
      
      // Error should be handled by component, not boundary
      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });
  });

  describe('Error Boundary with Nested Components', () => {
    it('should catch errors in nested components', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary>
            <div>
              <div>
                <ThrowingComponent shouldThrow={true} />
              </div>
            </div>
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    });

    it('should isolate errors to specific boundaries', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary>
            <div data-testid="outer-boundary">
              <ThrowingComponent shouldThrow={false} />
              <ErrorBoundary>
                <div data-testid="inner-boundary">
                  <ThrowingComponent shouldThrow={true} />
                </div>
              </ErrorBoundary>
            </div>
          </ErrorBoundary>
        </Wrapper>
      );
      
      // Inner boundary should catch error
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
      // Outer component should still render
      expect(screen.getByTestId('outer-boundary')).toBeInTheDocument();
    });
  });

  describe('Error Boundary with Async Operations', () => {
    it('should handle errors in async operations', async () => {
      const AsyncComponent = () => {
        const [error, setError] = React.useState<Error | null>(null);
        
        React.useEffect(() => {
          Promise.reject(new Error('Async error')).catch(setError);
        }, []);
        
        if (error) {
          throw error;
        }
        
        return <div>Loading...</div>;
      };
      
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary>
            <AsyncComponent />
          </ErrorBoundary>
        </Wrapper>
      );
      
      await waitFor(() => {
        expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
      });
    });
  });

  describe('Error Boundary Custom Fallback', () => {
    it('should use custom fallback when provided', () => {
      const customFallback = <div>Custom error message</div>;
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary fallback={customFallback}>
            <ThrowingComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(screen.getByText('Custom error message')).toBeInTheDocument();
    });
  });

  describe('Error Boundary Level', () => {
    it('should handle page-level errors', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary level="page">
            <ThrowingComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    });

    it('should handle component-level errors', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ErrorBoundary level="component">
            <ThrowingComponent shouldThrow={true} />
          </ErrorBoundary>
        </Wrapper>
      );
      
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    });
  });
});

