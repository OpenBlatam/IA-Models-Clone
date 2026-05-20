import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useApiHealth } from '@/lib/hooks/use-api-health';
import { checkApiHealth } from '@/lib/api/client';

// Mock the API client
jest.mock('@/lib/api/client', () => ({
  checkApiHealth: jest.fn(),
}));

const mockCheckApiHealth = checkApiHealth as jest.MockedFunction<
  typeof checkApiHealth
>;

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        cacheTime: 0,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useApiHealth', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return initial loading state', () => {
    mockCheckApiHealth.mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(
            () =>
              resolve({
                status: 'healthy',
                message: 'API is reachable',
                timestamp: Date.now(),
              }),
            100
          );
        })
    );

    const { result } = renderHook(() => useApiHealth(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.isHealthy).toBe(false);
    expect(result.current.message).toBe('Checking API status...');
  });

  it('should return healthy status when API is reachable', async () => {
    const timestamp = Date.now();
    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp,
    });

    const { result } = renderHook(() => useApiHealth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.isHealthy).toBe(true);
    expect(result.current.message).toBe('API is reachable');
    expect(result.current.lastChecked).toBe(timestamp);
    expect(result.current.error).toBeNull();
  });

  it('should return unhealthy status when API is unreachable', async () => {
    mockCheckApiHealth.mockResolvedValue({
      status: 'unhealthy',
      message: 'Connection failed',
      timestamp: Date.now(),
    });

    const { result } = renderHook(() => useApiHealth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.isHealthy).toBe(false);
    expect(result.current.message).toBe('Connection failed');
  });

  it('should handle errors', async () => {
    const error = new Error('Network error');
    mockCheckApiHealth.mockRejectedValue(error);

    const { result } = renderHook(() => useApiHealth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.isHealthy).toBe(false);
  });

  it('should not fetch when enabled is false', () => {
    renderHook(() => useApiHealth({ enabled: false }), {
      wrapper: createWrapper(),
    });

    // Should not call checkApiHealth immediately
    expect(mockCheckApiHealth).not.toHaveBeenCalled();
  });

  it('should use custom refetch interval', async () => {
    jest.useFakeTimers();
    const timestamp = Date.now();
    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp,
    });

    const { result } = renderHook(
      () => useApiHealth({ refetchInterval: 5000 }),
      {
        wrapper: createWrapper(),
      }
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Fast-forward time
    jest.advanceTimersByTime(5000);

    // Should refetch after interval
    await waitFor(() => {
      expect(mockCheckApiHealth).toHaveBeenCalledTimes(2);
    });

    jest.useRealTimers();
  });

  it('should refresh health check manually', async () => {
    const timestamp = Date.now();
    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp,
    });

    const { result } = renderHook(() => useApiHealth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const initialCallCount = mockCheckApiHealth.mock.calls.length;

    // Manually refresh
    result.current.refreshHealth();

    await waitFor(() => {
      expect(mockCheckApiHealth).toHaveBeenCalledTimes(initialCallCount + 1);
    });
  });

  it('should return default values when no data is available', () => {
    mockCheckApiHealth.mockImplementation(
      () =>
        new Promise((resolve) => {
          // Never resolve
        })
    );

    const { result } = renderHook(() => useApiHealth(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isHealthy).toBe(false);
    expect(result.current.lastChecked).toBeNull();
    expect(result.current.message).toBe('Checking API status...');
  });
});

