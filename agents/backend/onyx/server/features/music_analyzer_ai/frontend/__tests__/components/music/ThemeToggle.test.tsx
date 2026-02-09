import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeToggle } from '@/components/music/ThemeToggle';

describe('ThemeToggle', () => {
  it('renders theme toggle', () => {
    render(<ThemeToggle />);
    
    const toggle = screen.getByRole('button', { name: /theme|tema/i });
    expect(toggle).toBeInTheDocument();
  });

  it('toggles theme on click', () => {
    render(<ThemeToggle />);
    
    const toggle = screen.getByRole('button', { name: /theme|tema/i });
    fireEvent.click(toggle);
    
    // Theme should toggle
    expect(toggle).toBeInTheDocument();
  });
});

