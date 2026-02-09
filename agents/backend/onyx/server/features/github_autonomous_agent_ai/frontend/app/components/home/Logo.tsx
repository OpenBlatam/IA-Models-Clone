import { motion } from 'framer-motion';
import { clsx } from 'clsx';

interface LogoProps {
  size: 'sm' | 'lg';
  showText: boolean;
  gradientId: string;
}

export function Logo({ size, showText, gradientId }: LogoProps) {
  const iconSizeClasses = clsx({
    'w-6 h-6': size === 'sm',
    'w-14 h-14 md:w-16 md:h-16 lg:w-20 lg:h-20': size === 'lg',
  });

  const textSizeClasses = clsx({
    'text-base': size === 'sm',
    'text-3xl md:text-4xl lg:text-5xl': size === 'lg',
  });

  return (
    <div className={clsx("flex items-center", {
      "gap-2.5": size === 'sm',
      "gap-3": size === 'lg'
    })}>
      <div className={clsx(iconSizeClasses, "flex items-center justify-center flex-shrink-0")}>
        <svg
          viewBox="0 0 24 24"
          className="w-full h-full"
          aria-hidden="true"
          role="img"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#8800ff" />
              <stop offset="16.66%" stopColor="#0000ff" />
              <stop offset="33.33%" stopColor="#0088ff" />
              <stop offset="50%" stopColor="#00ff00" />
              <stop offset="66.66%" stopColor="#ffdd00" />
              <stop offset="83.33%" stopColor="#ff8800" />
              <stop offset="100%" stopColor="#ff0000" />
            </linearGradient>
          </defs>
          <path d="M7 20L12 4L17 20H14.5L12 12.5L9.5 20H7Z" fill={`url(#${gradientId})`} />
        </svg>
      </div>
      {showText && (
        <span className={clsx("font-normal text-black whitespace-nowrap", textSizeClasses)}>
          <span className="font-normal">bulk</span>
        </span>
      )}
    </div>
  );
}

