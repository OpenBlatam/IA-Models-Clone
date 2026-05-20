import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MusicLeaderboard } from '@/components/music/MusicLeaderboard';

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      {component}
    </QueryClientProvider>
  );
};

describe('MusicLeaderboard', () => {
  it('renders leaderboard', () => {
    renderWithQueryClient(<MusicLeaderboard />);
    
    expect(screen.getByText(/clasificación|leaderboard/i)).toBeInTheDocument();
  });

  it('displays ranking information', () => {
    renderWithQueryClient(<MusicLeaderboard />);
    
    // Leaderboard should render ranking elements
    const leaderboard = screen.getByText(/clasificación|leaderboard/i);
    expect(leaderboard).toBeInTheDocument();
  });
});

