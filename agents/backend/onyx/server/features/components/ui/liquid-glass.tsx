import React from 'react';
import { cn } from '@/lib/utils';

interface LiquidGlassProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'button' | 'dock' | 'menu';
  intensity?: number;
  blur?: number;
  shine?: boolean;
  interactive?: boolean;
}

export function LiquidGlass({ 
  children, 
  className, 
  variant = 'default',
  intensity = 1,
  blur = 6,
  shine = true,
  interactive = true,
  ...props 
}: LiquidGlassProps) {
  const baseStyles = "relative flex font-semibold overflow-hidden text-black cursor-pointer transition-all duration-400 ease-[cubic-bezier(0.175,0.885,0.32,2.2)]";
  
  const variantStyles = {
    default: "rounded-2xl",
    button: "rounded-[3rem] p-[1.5rem_2.5rem]",
    dock: "rounded-[2rem] p-[0.6rem]",
    menu: "rounded-[1.8rem] p-[0.4rem]"
  };

  const variantHoverStyles = {
    default: "hover:rounded-[2.5rem]",
    button: "hover:rounded-[4rem] hover:p-[1.8rem_2.8rem]",
    dock: "hover:rounded-[2.5rem] hover:p-[0.8rem]",
    menu: "hover:rounded-[1.8rem] hover:p-[0.6rem]"
  };

  return (
    <div
      className={cn(
        baseStyles,
        variantStyles[variant],
        variantHoverStyles[variant],
        "shadow-[0_8px_32px_0_rgba(31,38,135,0.15)]",
        "before:absolute before:inset-0 before:bg-gradient-to-b before:from-white/40 before:to-white/15",
        "after:absolute after:inset-0 after:bg-gradient-to-b after:from-white/15 after:to-transparent",
        "border border-white/20",
        className
      )}
      {...props}
    >
      {/* Effect Layer */}
      <div className="absolute inset-0 z-0 overflow-hidden isolation-auto backdrop-blur-[6px]">
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          <defs>
            <filter id="glass-distortion">
              <feTurbulence
                type="fractalNoise"
                baseFrequency={0.006 * intensity}
                numOctaves="3"
                result="noise"
              />
              <feDisplacementMap
                in="SourceGraphic"
                in2="noise"
                scale={12 * intensity}
                xChannelSelector="R"
                yChannelSelector="G"
              />
            </filter>
            <filter id="glass-blur">
              <feGaussianBlur stdDeviation="0.6" />
            </filter>
            <filter id="glass-highlight">
              <feGaussianBlur stdDeviation="0.2" />
              <feColorMatrix
                type="matrix"
                values="1 0 0 0 1
                        0 1 0 0 1
                        0 0 1 0 1
                        0 0 0 12 -4"
              />
            </filter>
          </defs>
        </svg>
      </div>

      {/* Tint Layer */}
      <div className="absolute inset-0 z-1 bg-gradient-to-b from-white/20 to-white/5" />

      {/* Shine Layer */}
      {shine && (
        <div className="absolute inset-0 z-2 overflow-hidden">
          <div className="absolute inset-0 shadow-[inset_1px_1px_1px_0_rgba(255,255,255,0.4),inset_-1px_-1px_1px_0_rgba(255,255,255,0.2)]" />
        </div>
      )}

      {/* Content Layer */}
      <div className="relative z-3">
        {children}
      </div>
    </div>
  );
} 