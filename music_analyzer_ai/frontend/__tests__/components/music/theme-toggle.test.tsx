import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeToggle } from '@/components/music/ThemeToggle';

describe('ThemeToggle', () => {
  beforeEach(() => {
    // Reset document classes
    document.documentElement.classList.remove('dark');
  });

  it('should render toggle button', () => {
    render(<ThemeToggle />);

    const button = screen.getByRole('button', { name: /toggle theme/i });
    expect(button).toBeInTheDocument();
  });

  it('should show sun icon when dark mode is active', () => {
    render(<ThemeToggle />);

    // Component starts with dark mode
    const sunIcon = screen.getByRole('button').querySelector('svg');
    expect(sunIcon).toBeInTheDocument();
  });

  it('should toggle theme when clicked', async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);

    const button = screen.getByRole('button', { name: /toggle theme/i });

    // Initially dark mode
    expect(document.documentElement.classList.contains('dark')).toBe(true);

    // Toggle to light mode
    await user.click(button);

    expect(document.documentElement.classList.contains('dark')).toBe(false);
  });

  it('should add dark class when isDark is true', () => {
    render(<ThemeToggle />);

    // Component initializes with dark mode
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });

  it('should remove dark class when isDark is false', async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);

    const button = screen.getByRole('button', { name: /toggle theme/i });

    await user.click(button);

    expect(document.documentElement.classList.contains('dark')).toBe(false);
  });

  it('should have correct aria-label', () => {
    render(<ThemeToggle />);

    const button = screen.getByRole('button', { name: /toggle theme/i });
    expect(button).toHaveAttribute('aria-label', 'Toggle theme');
  });

  it('should have hover styles', () => {
    const { container } = render(<ThemeToggle />);

    const button = container.querySelector('button');
    expect(button?.className).toContain('hover:bg-white/20');
  });
});

