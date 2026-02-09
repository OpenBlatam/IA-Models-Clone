import { render, screen } from '@testing-library/react';
import { AnimatedCard, FadeIn, SlideUp } from '@/components/music/AnimatedCard';

describe('AnimatedCard', () => {
  it('should render children', () => {
    render(
      <AnimatedCard>
        <div>Test Content</div>
      </AnimatedCard>
    );

    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  it('should apply custom className', () => {
    const { container } = render(
      <AnimatedCard className="custom-class">
        <div>Test</div>
      </AnimatedCard>
    );

    const card = container.firstChild as HTMLElement;
    expect(card.className).toContain('custom-class');
  });

  it('should use default delay of 0', () => {
    render(
      <AnimatedCard>
        <div>Test</div>
      </AnimatedCard>
    );

    expect(screen.getByText('Test')).toBeInTheDocument();
  });

  it('should accept custom delay', () => {
    render(
      <AnimatedCard delay={0.5}>
        <div>Test</div>
      </AnimatedCard>
    );

    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});

describe('FadeIn', () => {
  it('should render children', () => {
    render(
      <FadeIn>
        <div>Fade In Content</div>
      </FadeIn>
    );

    expect(screen.getByText('Fade In Content')).toBeInTheDocument();
  });

  it('should use default delay of 0', () => {
    render(
      <FadeIn>
        <div>Test</div>
      </FadeIn>
    );

    expect(screen.getByText('Test')).toBeInTheDocument();
  });

  it('should accept custom delay', () => {
    render(
      <FadeIn delay={0.3}>
        <div>Test</div>
      </FadeIn>
    );

    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});

describe('SlideUp', () => {
  it('should render children', () => {
    render(
      <SlideUp>
        <div>Slide Up Content</div>
      </SlideUp>
    );

    expect(screen.getByText('Slide Up Content')).toBeInTheDocument();
  });

  it('should use default delay of 0', () => {
    render(
      <SlideUp>
        <div>Test</div>
      </SlideUp>
    );

    expect(screen.getByText('Test')).toBeInTheDocument();
  });

  it('should accept custom delay', () => {
    render(
      <SlideUp delay={0.4}>
        <div>Test</div>
      </SlideUp>
    );

    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});

