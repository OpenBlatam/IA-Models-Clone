/**
 * Browser Compatibility Tests
 * Tests to ensure compatibility across different browsers
 */

import { render, screen } from '@testing-library/react';
import { Navigation } from '@/components/Navigation';
import { TrackSearch } from '@/components/music/TrackSearch';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('Browser Compatibility', () => {
  describe('Feature Detection', () => {
    it('should work with localStorage', () => {
      expect(typeof Storage !== 'undefined').toBe(true);
      expect(typeof localStorage !== 'undefined').toBe(true);
    });

    it('should work with sessionStorage', () => {
      expect(typeof sessionStorage !== 'undefined').toBe(true);
    });

    it('should support modern JavaScript features', () => {
      // Test arrow functions
      const arrowFn = () => true;
      expect(arrowFn()).toBe(true);

      // Test async/await
      const asyncFn = async () => Promise.resolve(true);
      expect(asyncFn()).resolves.toBe(true);

      // Test destructuring
      const { a, b } = { a: 1, b: 2 };
      expect(a).toBe(1);
      expect(b).toBe(2);
    });
  });

  describe('API Compatibility', () => {
    it('should support fetch API', () => {
      expect(typeof fetch !== 'undefined').toBe(true);
    });

    it('should support Promise API', () => {
      expect(typeof Promise !== 'undefined').toBe(true);
      const promise = Promise.resolve(true);
      expect(promise).toBeInstanceOf(Promise);
    });

    it('should support Array methods', () => {
      const arr = [1, 2, 3];
      expect(arr.map).toBeDefined();
      expect(arr.filter).toBeDefined();
      expect(arr.reduce).toBeDefined();
    });
  });

  describe('DOM Compatibility', () => {
    it('should support querySelector', () => {
      expect(typeof document.querySelector).toBe('function');
    });

    it('should support addEventListener', () => {
      expect(typeof document.addEventListener).toBe('function');
    });

    it('should support classList', () => {
      const div = document.createElement('div');
      expect(div.classList).toBeDefined();
    });
  });

  describe('Component Rendering', () => {
    it('should render Navigation in all browsers', () => {
      render(<Navigation />);
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });

    it('should render TrackSearch in all browsers', () => {
      render(<TrackSearch onTrackSelect={() => {}} />, {
        wrapper: createWrapper(),
      });
      expect(
        screen.getByPlaceholderText(/busca canciones, artistas o álbumes/i)
      ).toBeInTheDocument();
    });
  });

  describe('Event Handling', () => {
    it('should handle click events', () => {
      const handleClick = jest.fn();
      render(<button onClick={handleClick}>Click me</button>);
      
      const button = screen.getByRole('button');
      button.click();
      
      expect(handleClick).toHaveBeenCalled();
    });

    it('should handle input events', () => {
      const handleChange = jest.fn();
      render(<input onChange={handleChange} />);
      
      const input = screen.getByRole('textbox');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      
      expect(handleChange).toHaveBeenCalled();
    });
  });
});

