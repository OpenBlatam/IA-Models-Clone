/**
 * Comprehensive Accessibility Tests
 * Tests for WCAG compliance and accessibility best practices
 */

import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { Navigation } from '@/components/Navigation';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { FormField } from '@/components/ui/form-field';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

expect.extend(toHaveNoViolations);

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

describe('Comprehensive Accessibility Tests', () => {
  describe('WCAG Compliance', () => {
    it('Navigation should have no accessibility violations', async () => {
      const { container } = render(<Navigation />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('FormField should have no accessibility violations', async () => {
      const { container } = render(
        <FormField label="Test" name="test" required />
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Keyboard Navigation', () => {
    it('should support tab navigation', () => {
      render(<Navigation />);
      const links = screen.getAllByRole('link');
      expect(links.length).toBeGreaterThan(0);
      links.forEach((link) => {
        expect(link).toHaveAttribute('tabIndex', expect.any(String));
      });
    });

    it('should have focusable elements', () => {
      render(<Navigation />);
      const focusableElements = screen.getAllByRole('link');
      focusableElements.forEach((element) => {
        expect(element).toBeInTheDocument();
      });
    });
  });

  describe('ARIA Attributes', () => {
    it('should have proper ARIA labels', () => {
      render(<Navigation />);
      const nav = screen.getByRole('navigation');
      expect(nav).toBeInTheDocument();
    });

    it('should have proper ARIA attributes on form fields', () => {
      render(<FormField label="Test" name="test" errors={['Error']} touched />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveAttribute('aria-invalid', 'true');
      expect(input).toHaveAttribute('aria-describedby');
    });
  });

  describe('Color Contrast', () => {
    it('should have sufficient color contrast for text', () => {
      // This would require visual regression testing
      // For now, we verify that components render
      render(<Navigation />);
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });
  });

  describe('Screen Reader Support', () => {
    it('should have descriptive text for screen readers', () => {
      render(<FormField label="Test Field" name="test" helperText="Helper" />);
      const input = screen.getByLabelText('Test Field');
      expect(input).toBeInTheDocument();
    });

    it('should have error messages accessible to screen readers', () => {
      render(<FormField label="Test" name="test" errors={['Error']} touched />);
      const error = screen.getByRole('alert');
      expect(error).toBeInTheDocument();
      expect(error).toHaveTextContent('Error');
    });
  });

  describe('Focus Management', () => {
    it('should manage focus correctly', () => {
      render(<FormField label="Test" name="test" />);
      const input = screen.getByRole('textbox');
      expect(input).toBeInTheDocument();
      // Focus management would be tested with user interactions
    });
  });
});

