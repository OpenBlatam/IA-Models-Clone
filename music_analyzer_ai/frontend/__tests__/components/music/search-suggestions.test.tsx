import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SearchSuggestions } from '@/components/music/SearchSuggestions';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

describe('SearchSuggestions', () => {
  const mockOnSelect = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.clear();
  });

  it('should render trending searches', () => {
    render(<SearchSuggestions onSelect={mockOnSelect} />);

    expect(screen.getByText('Tendencias')).toBeInTheDocument();
    expect(screen.getByText('Bohemian Rhapsody')).toBeInTheDocument();
    expect(screen.getByText('Blinding Lights')).toBeInTheDocument();
    expect(screen.getByText('Shape of You')).toBeInTheDocument();
  });

  it('should not show recent searches when localStorage is empty', () => {
    render(<SearchSuggestions onSelect={mockOnSelect} />);

    expect(screen.queryByText('Búsquedas Recientes')).not.toBeInTheDocument();
  });

  it('should show recent searches from localStorage', () => {
    localStorageMock.setItem(
      'recent-searches',
      JSON.stringify(['Recent Search 1', 'Recent Search 2'])
    );

    render(<SearchSuggestions onSelect={mockOnSelect} />);

    expect(screen.getByText('Búsquedas Recientes')).toBeInTheDocument();
    expect(screen.getByText('Recent Search 1')).toBeInTheDocument();
    expect(screen.getByText('Recent Search 2')).toBeInTheDocument();
  });

  it('should call onSelect when trending search is clicked', async () => {
    const user = userEvent.setup();
    render(<SearchSuggestions onSelect={mockOnSelect} />);

    const trendingButton = screen.getByText('Bohemian Rhapsody');
    await user.click(trendingButton);

    expect(mockOnSelect).toHaveBeenCalledWith('Bohemian Rhapsody');
  });

  it('should call onSelect when recent search is clicked', async () => {
    const user = userEvent.setup();
    localStorageMock.setItem(
      'recent-searches',
      JSON.stringify(['My Recent Search'])
    );

    render(<SearchSuggestions onSelect={mockOnSelect} />);

    const recentButton = screen.getByText('My Recent Search');
    await user.click(recentButton);

    expect(mockOnSelect).toHaveBeenCalledWith('My Recent Search');
  });

  it('should save selected search to localStorage', async () => {
    const user = userEvent.setup();
    render(<SearchSuggestions onSelect={mockOnSelect} />);

    const trendingButton = screen.getByText('Bohemian Rhapsody');
    await user.click(trendingButton);

    await waitFor(() => {
      const saved = localStorageMock.getItem('recent-searches');
      expect(saved).toBeTruthy();
      if (saved) {
        const parsed = JSON.parse(saved);
        expect(parsed).toContain('Bohemian Rhapsody');
      }
    });
  });

  it('should limit recent searches to 5', async () => {
    const user = userEvent.setup();
    localStorageMock.setItem(
      'recent-searches',
      JSON.stringify(['1', '2', '3', '4', '5'])
    );

    render(<SearchSuggestions onSelect={mockOnSelect} />);

    const trendingButton = screen.getByText('Bohemian Rhapsody');
    await user.click(trendingButton);

    await waitFor(() => {
      const saved = localStorageMock.getItem('recent-searches');
      if (saved) {
        const parsed = JSON.parse(saved);
        expect(parsed.length).toBeLessThanOrEqual(5);
      }
    });
  });

  it('should move selected search to top of recent searches', async () => {
    const user = userEvent.setup();
    localStorageMock.setItem(
      'recent-searches',
      JSON.stringify(['Old Search 1', 'Old Search 2'])
    );

    render(<SearchSuggestions onSelect={mockOnSelect} />);

    const trendingButton = screen.getByText('Bohemian Rhapsody');
    await user.click(trendingButton);

    await waitFor(() => {
      const saved = localStorageMock.getItem('recent-searches');
      if (saved) {
        const parsed = JSON.parse(saved);
        expect(parsed[0]).toBe('Bohemian Rhapsody');
      }
    });
  });

  it('should remove duplicate from recent searches', async () => {
    const user = userEvent.setup();
    localStorageMock.setItem(
      'recent-searches',
      JSON.stringify(['Bohemian Rhapsody', 'Other Search'])
    );

    render(<SearchSuggestions onSelect={mockOnSelect} />);

    const trendingButton = screen.getByText('Bohemian Rhapsody');
    await user.click(trendingButton);

    await waitFor(() => {
      const saved = localStorageMock.getItem('recent-searches');
      if (saved) {
        const parsed = JSON.parse(saved);
        const count = parsed.filter(
          (s: string) => s === 'Bohemian Rhapsody'
        ).length;
        expect(count).toBe(1);
      }
    });
  });

  it('should handle invalid localStorage data gracefully', () => {
    localStorageMock.setItem('recent-searches', 'invalid-json');

    // Should not crash
    render(<SearchSuggestions onSelect={mockOnSelect} />);

    expect(screen.getByText('Tendencias')).toBeInTheDocument();
  });
});

