"use client";

import React from "react";
import { cn } from "../../utils/classNames";

type AnimatedContainerProps = {
  readonly children: React.ReactNode;
  readonly className?: string;
  readonly animation?: "fade" | "slide" | "scale";
  readonly delay?: number;
};

export const AnimatedContainer = ({
  children,
  className,
  animation = "fade",
  delay = 0,
}: AnimatedContainerProps): JSX.Element => {
  const animationClasses = {
    fade: "animate-in fade-in duration-300",
    slide: "animate-in slide-in-from-bottom-4 duration-300",
    scale: "animate-in zoom-in-95 duration-300",
  };

  return (
    <div
      className={cn(animationClasses[animation], className)}
      style={{ animationDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
};







