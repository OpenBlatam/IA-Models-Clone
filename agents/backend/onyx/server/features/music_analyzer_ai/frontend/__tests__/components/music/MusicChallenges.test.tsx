import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MusicChallenges } from '@/components/music/MusicChallenges';

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

describe('MusicChallenges', () => {
  it('renders challenges', () => {
    renderWithQueryClient(<MusicChallenges />);
    
    expect(screen.getByText(/desafíos|challenges/i)).toBeInTheDocument();
  });

  it('displays challenge information', () => {
    renderWithQueryClient(<MusicChallenges />);
    
    const challenges = screen.getByText(/desafíos|challenges/i);
    expect(challenges).toBeInTheDocument();
  });
});

