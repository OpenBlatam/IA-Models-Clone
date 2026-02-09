import { render, screen } from '@testing-library/react';
import { Navigation } from '@/components/Navigation';

// Mock next/navigation
const mockPush = jest.fn();
const mockPathname = jest.fn();

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: jest.fn(),
    prefetch: jest.fn(),
    back: jest.fn(),
  }),
  usePathname: () => mockPathname(),
  useSearchParams: () => new URLSearchParams(),
}));

describe('Navigation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render navigation links', () => {
    mockPathname.mockReturnValue('/');
    render(<Navigation />);

    expect(screen.getByText('Inicio')).toBeInTheDocument();
    expect(screen.getByText('Music AI')).toBeInTheDocument();
    expect(screen.getByText('Robot AI')).toBeInTheDocument();
  });

  it('should highlight active link', () => {
    mockPathname.mockReturnValue('/music');
    render(<Navigation />);

    const musicLink = screen.getByText('Music AI').closest('a');
    expect(musicLink?.className).toContain('bg-white/20');
  });

  it('should not highlight inactive links', () => {
    mockPathname.mockReturnValue('/music');
    render(<Navigation />);

    const homeLink = screen.getByText('Inicio').closest('a');
    expect(homeLink?.className).not.toContain('bg-white/20');
    expect(homeLink?.className).toContain('text-gray-300');
  });

  it('should render brand name', () => {
    mockPathname.mockReturnValue('/');
    render(<Navigation />);

    expect(screen.getByText('Blatam Academy')).toBeInTheDocument();
  });

  it('should have correct hrefs for navigation links', () => {
    mockPathname.mockReturnValue('/');
    render(<Navigation />);

    const homeLink = screen.getByText('Inicio').closest('a');
    const musicLink = screen.getByText('Music AI').closest('a');
    const robotLink = screen.getByText('Robot AI').closest('a');

    expect(homeLink).toHaveAttribute('href', '/');
    expect(musicLink).toHaveAttribute('href', '/music');
    expect(robotLink).toHaveAttribute('href', '/robot');
  });

  it('should highlight home link when on home page', () => {
    mockPathname.mockReturnValue('/');
    render(<Navigation />);

    const homeLink = screen.getByText('Inicio').closest('a');
    expect(homeLink?.className).toContain('bg-white/20');
  });

  it('should highlight robot link when on robot page', () => {
    mockPathname.mockReturnValue('/robot');
    render(<Navigation />);

    const robotLink = screen.getByText('Robot AI').closest('a');
    expect(robotLink?.className).toContain('bg-white/20');
  });
});

