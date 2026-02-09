import { memo, useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';

interface ConfettiProps {
  active: boolean;
  duration?: number;
  particleCount?: number;
  className?: string;
}

const Confetti = memo(({
  active,
  duration = 3000,
  particleCount = 50,
  className = '',
}: ConfettiProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!active || !containerRef.current) {
      return;
    }

    const container = containerRef.current;
    const particles: HTMLDivElement[] = [];
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'];

    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div');
      const size = Math.random() * 10 + 5;
      const color = colors[Math.floor(Math.random() * colors.length)];

      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.backgroundColor = color;
      particle.style.position = 'fixed';
      particle.style.left = `${Math.random() * 100}%`;
      particle.style.top = '-10px';
      particle.style.borderRadius = '50%';
      particle.style.pointerEvents = 'none';
      particle.style.zIndex = '9999';
      particle.style.animation = `confetti-fall ${duration}ms linear forwards`;
      particle.style.animationDelay = `${Math.random() * 1000}ms`;

      container.appendChild(particle);
      particles.push(particle);
    }

    const style = document.createElement('style');
    style.textContent = `
      @keyframes confetti-fall {
        to {
          transform: translateY(100vh) rotate(360deg);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);

    const timeout = setTimeout(() => {
      particles.forEach((particle) => particle.remove());
      style.remove();
    }, duration);

    return () => {
      clearTimeout(timeout);
      particles.forEach((particle) => particle.remove());
      style.remove();
    };
  }, [active, duration, particleCount]);

  if (!active) {
    return null;
  }

  return <div ref={containerRef} className={cn('fixed inset-0 pointer-events-none', className)} />;
});

Confetti.displayName = 'Confetti';

export default Confetti;



