import { render, screen } from '@testing-library/react';
import { LoadingSkeleton, TrackCardSkeleton } from '@/components/music/LoadingSkeleton';

describe('LoadingSkeleton', () => {
  it('should render loading skeleton', () => {
    render(<LoadingSkeleton />);

    const skeleton = screen.getByRole('generic');
    expect(skeleton).toBeInTheDocument();
    expect(skeleton.className).toContain('animate-pulse');
  });

  it('should have correct structure with multiple skeleton elements', () => {
    const { container } = render(<LoadingSkeleton />);

    const skeletonElements = container.querySelectorAll('.bg-white\\/10');
    expect(skeletonElements.length).toBeGreaterThan(0);
  });

  it('should have pulse animation', () => {
    const { container } = render(<LoadingSkeleton />);

    const skeleton = container.firstChild as HTMLElement;
    expect(skeleton.className).toContain('animate-pulse');
  });
});

describe('TrackCardSkeleton', () => {
  it('should render track card skeleton', () => {
    render(<TrackCardSkeleton />);

    const skeleton = screen.getByRole('generic');
    expect(skeleton).toBeInTheDocument();
    expect(skeleton.className).toContain('animate-pulse');
  });

  it('should have correct layout structure', () => {
    const { container } = render(<TrackCardSkeleton />);

    const skeleton = container.firstChild as HTMLElement;
    expect(skeleton.className).toContain('flex');
    expect(skeleton.className).toContain('items-center');
  });

  it('should have image placeholder', () => {
    const { container } = render(<TrackCardSkeleton />);

    const imagePlaceholder = container.querySelector('.w-12.h-12');
    expect(imagePlaceholder).toBeInTheDocument();
  });

  it('should have text placeholders', () => {
    const { container } = render(<TrackCardSkeleton />);

    const textPlaceholders = container.querySelectorAll('.bg-white\\/10');
    expect(textPlaceholders.length).toBeGreaterThan(0);
  });
});

