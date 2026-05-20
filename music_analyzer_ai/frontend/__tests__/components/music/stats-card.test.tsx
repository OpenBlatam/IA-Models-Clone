import { render, screen } from '@testing-library/react';
import { StatsCard } from '@/components/music/StatsCard';
import { Music } from 'lucide-react';

describe('StatsCard', () => {
  it('should render title and value', () => {
    render(<StatsCard title="Test Title" value="100" />);

    expect(screen.getByText('Test Title')).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();
  });

  it('should render numeric value', () => {
    render(<StatsCard title="Count" value={42} />);

    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it('should render icon when provided', () => {
    render(<StatsCard title="Test" value="100" icon={<Music data-testid="icon" />} />);

    expect(screen.getByTestId('icon')).toBeInTheDocument();
  });

  it('should show trend up with green color', () => {
    render(<StatsCard title="Test" value="100" change={10} trend="up" />);

    const trendIcon = screen.getByText('10%').closest('div');
    expect(trendIcon?.className).toContain('text-green-400');
  });

  it('should show trend down with red color', () => {
    render(<StatsCard title="Test" value="100" change={5} trend="down" />);

    const trendIcon = screen.getByText('5%').closest('div');
    expect(trendIcon?.className).toContain('text-red-400');
  });

  it('should show neutral trend with gray color', () => {
    render(<StatsCard title="Test" value="100" change={0} trend="neutral" />);

    const trendIcon = screen.getByText('0%').closest('div');
    expect(trendIcon?.className).toContain('text-gray-400');
  });

  it('should display absolute value of change', () => {
    render(<StatsCard title="Test" value="100" change={-15} trend="down" />);

    expect(screen.getByText('15%')).toBeInTheDocument();
  });

  it('should not show change when not provided', () => {
    render(<StatsCard title="Test" value="100" />);

    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });

  it('should have hover effect', () => {
    const { container } = render(<StatsCard title="Test" value="100" />);

    const card = container.firstChild as HTMLElement;
    expect(card.className).toContain('hover:bg-white/15');
    expect(card.className).toContain('transition-colors');
  });

  it('should handle large values', () => {
    render(<StatsCard title="Large" value="1,000,000" />);

    expect(screen.getByText('1,000,000')).toBeInTheDocument();
  });

  it('should handle zero value', () => {
    render(<StatsCard title="Zero" value={0} />);

    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('should handle negative change with trend', () => {
    render(<StatsCard title="Test" value="50" change={-20} trend="down" />);

    expect(screen.getByText('20%')).toBeInTheDocument();
    const trendIcon = screen.getByText('20%').closest('div');
    expect(trendIcon?.className).toContain('text-red-400');
  });

  it('should render correctly with all props', () => {
    render(
      <StatsCard
        title="Complete Card"
        value="500"
        change={25}
        trend="up"
        icon={<Music data-testid="complete-icon" />}
      />
    );

    expect(screen.getByText('Complete Card')).toBeInTheDocument();
    expect(screen.getByText('500')).toBeInTheDocument();
    expect(screen.getByText('25%')).toBeInTheDocument();
    expect(screen.getByTestId('complete-icon')).toBeInTheDocument();
  });
});

