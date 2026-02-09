import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiStatus } from '@/components/api-status';
import { checkApiHealth } from '@/lib/api/client';

// Mock the API client
jest.mock('@/lib/api/client', () => ({
  checkApiHealth: jest.fn(),
}));

const mockCheckApiHealth = checkApiHealth as jest.MockedFunction<
  typeof checkApiHealth
>;

describe('ApiStatus', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
          cacheTime: 0,
        },
      },
    });
    jest.clearAllMocks();
  });

  const renderWithProvider = (props = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <ApiStatus {...props} />
      </QueryClientProvider>
    );
  };

  it('should render loading state initially', () => {
    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp: Date.now(),
    });

    renderWithProvider();
    expect(screen.getByText(/checking/i)).toBeInTheDocument();
  });

  it('should display connected state when API is healthy', async () => {
    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp: Date.now(),
    });

    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });

  it('should display disconnected state when API is unhealthy', async () => {
    mockCheckApiHealth.mockResolvedValue({
      status: 'unhealthy',
      message: 'Connection failed',
      timestamp: Date.now(),
    });

    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });
  });

  it('should show detailed status when showDetails is true', async () => {
    const timestamp = Date.now();
    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp,
    });

    renderWithProvider({ showDetails: true });

    await waitFor(() => {
      expect(screen.getByText('API Status')).toBeInTheDocument();
      expect(screen.getByText('API is reachable')).toBeInTheDocument();
    });
  });

  it('should show error message when API is unhealthy with details', async () => {
    mockCheckApiHealth.mockResolvedValue({
      status: 'unhealthy',
      message: 'Connection failed',
      timestamp: Date.now(),
    });

    renderWithProvider({ showDetails: true });

    await waitFor(() => {
      expect(
        screen.getByText(/unable to connect to the api/i)
      ).toBeInTheDocument();
    });
  });

  it('should call refreshHealth when refresh button is clicked', async () => {
    const user = userEvent.setup();
    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp: Date.now(),
    });

    renderWithProvider();

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    const refreshButton = screen.getByTitle(/refresh/i);
    await user.click(refreshButton);

    expect(mockCheckApiHealth).toHaveBeenCalledTimes(2); // Initial + refresh
  });

  it('should apply correct position classes', () => {
    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp: Date.now(),
    });

    const { container } = renderWithProvider({ position: 'top-left' });
    const statusElement = container.firstChild as HTMLElement;
    expect(statusElement.className).toContain('top-4');
    expect(statusElement.className).toContain('left-4');
  });

  it('should disable refresh button while loading', async () => {
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

    renderWithProvider();

    const refreshButton = screen.getByTitle(/refresh|checking/i);
    expect(refreshButton).toBeDisabled();
  });
});

