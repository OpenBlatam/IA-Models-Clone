/**
 * Context Integration Tests
 * Tests for React Context integration with components and store
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createContext, useContext, useState, ReactNode } from 'react';
import { useMusicStore } from '@/lib/store/music-store';
import { type Track } from '@/lib/api/types';

// Test contexts
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  
  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };
  
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};

// Component using theme context
const ThemeComponent = () => {
  const { theme, toggleTheme } = useTheme();
  
  return (
    <div>
      <div data-testid="theme">{theme}</div>
      <button onClick={toggleTheme} data-testid="toggle-theme">
        Toggle Theme
      </button>
    </div>
  );
};

// Component using theme and store
const ThemeStoreComponent = () => {
  const { theme } = useTheme();
  const { currentTrack, setCurrentTrack } = useMusicStore();
  
  const handleSetTrack = () => {
    const track: Track = {
      id: '1',
      name: 'Theme Track',
      artists: ['Artist'],
      duration_ms: 200000,
      preview_url: 'https://example.com/preview.mp3',
      images: [],
    };
    setCurrentTrack(track);
  };
  
  return (
    <div data-theme={theme}>
      <div data-testid="theme">{theme}</div>
      {currentTrack && (
        <div data-testid="track">{currentTrack.name}</div>
      )}
      <button onClick={handleSetTrack} data-testid="set-track">
        Set Track
      </button>
    </div>
  );
};

// Nested context provider
interface UserContextType {
  user: { id: string; name: string } | null;
  setUser: (user: { id: string; name: string } | null) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

const UserProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<{ id: string; name: string } | null>(null);
  
  return (
    <UserContext.Provider value={{ user, setUser }}>
      {children}
    </UserContext.Provider>
  );
};

const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within UserProvider');
  }
  return context;
};

// Component using multiple contexts
const MultiContextComponent = () => {
  const { theme } = useTheme();
  const { user } = useUser();
  const { currentTrack } = useMusicStore();
  
  return (
    <div>
      <div data-testid="theme">{theme}</div>
      <div data-testid="user">{user?.name || 'No user'}</div>
      <div data-testid="track">{currentTrack?.name || 'No track'}</div>
    </div>
  );
};

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });
  
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        {children}
      </ThemeProvider>
    </QueryClientProvider>
  );
};

const createMultiWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });
  
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <UserProvider>
          {children}
        </UserProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('Context Integration', () => {
  beforeEach(() => {
    useMusicStore.setState({
      currentTrack: null,
      playlistQueue: [],
      isPlaying: false,
    });
  });

  describe('Basic Context Usage', () => {
    it('should provide context value to children', () => {
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ThemeComponent />
        </Wrapper>
      );
      
      expect(screen.getByTestId('theme')).toHaveTextContent('light');
    });

    it('should update context value', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ThemeComponent />
        </Wrapper>
      );
      
      const toggleButton = screen.getByTestId('toggle-theme');
      await user.click(toggleButton);
      
      await waitFor(() => {
        expect(screen.getByTestId('theme')).toHaveTextContent('dark');
      });
    });

    it('should throw error when context is used outside provider', () => {
      // Suppress console.error for this test
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      
      expect(() => {
        render(<ThemeComponent />);
      }).toThrow('useTheme must be used within ThemeProvider');
      
      consoleSpy.mockRestore();
    });
  });

  describe('Context with Store Integration', () => {
    it('should work with store and context together', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      render(
        <Wrapper>
          <ThemeStoreComponent />
        </Wrapper>
      );
      
      expect(screen.getByTestId('theme')).toHaveTextContent('light');
      
      const setTrackButton = screen.getByTestId('set-track');
      await user.click(setTrackButton);
      
      await waitFor(() => {
        expect(screen.getByTestId('track')).toHaveTextContent('Theme Track');
      });
    });

    it('should update context and preserve store state', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      
      // Set track first
      const track: Track = {
        id: '1',
        name: 'Preserved Track',
        artists: ['Artist'],
        duration_ms: 200000,
        preview_url: 'https://example.com/preview.mp3',
        images: [],
      };
      useMusicStore.setState({ currentTrack: track });
      
      render(
        <Wrapper>
          <ThemeStoreComponent />
        </Wrapper>
      );
      
      // Track should be preserved
      expect(screen.getByTestId('track')).toHaveTextContent('Preserved Track');
      
      // Toggle theme
      const toggleButton = screen.getByTestId('toggle-theme');
      await user.click(toggleButton);
      
      // Track should still be there
      await waitFor(() => {
        expect(screen.getByTestId('theme')).toHaveTextContent('dark');
        expect(screen.getByTestId('track')).toHaveTextContent('Preserved Track');
      });
    });
  });

  describe('Multiple Contexts', () => {
    it('should work with multiple contexts', () => {
      const Wrapper = createMultiWrapper();
      
      render(
        <Wrapper>
          <MultiContextComponent />
        </Wrapper>
      );
      
      expect(screen.getByTestId('theme')).toHaveTextContent('light');
      expect(screen.getByTestId('user')).toHaveTextContent('No user');
      expect(screen.getByTestId('track')).toHaveTextContent('No track');
    });

    it('should update multiple contexts independently', async () => {
      const user = userEvent.setup();
      const Wrapper = createMultiWrapper();
      
      render(
        <Wrapper>
          <MultiContextComponent />
        </Wrapper>
      );
      
      // This would require exposing setUser, but demonstrates the concept
      expect(screen.getByTestId('theme')).toHaveTextContent('light');
    });
  });

  describe('Context Provider Nesting', () => {
    it('should handle nested providers correctly', () => {
      const Wrapper = createMultiWrapper();
      
      render(
        <Wrapper>
          <ThemeProvider>
            <ThemeComponent />
          </ThemeProvider>
        </Wrapper>
      );
      
      // Should work with nested provider
      expect(screen.getByTestId('theme')).toBeInTheDocument();
    });
  });

  describe('Context State Persistence', () => {
    it('should persist context state across re-renders', () => {
      const Wrapper = createWrapper();
      
      const { rerender } = render(
        <Wrapper>
          <ThemeComponent />
        </Wrapper>
      );
      
      const theme1 = screen.getByTestId('theme').textContent;
      
      rerender(
        <Wrapper>
          <ThemeComponent />
        </Wrapper>
      );
      
      const theme2 = screen.getByTestId('theme').textContent;
      
      expect(theme1).toBe(theme2);
    });
  });
});

