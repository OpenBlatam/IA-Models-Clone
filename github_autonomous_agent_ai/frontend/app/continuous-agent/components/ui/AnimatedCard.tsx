"use client";

import React from "react";
import { motion } from "framer-motion";
import { cardAnimations } from "../../constants/animations";
import { cn } from "../../utils/classNames";
import type { ReactNode } from "react";

type AnimatedCardProps = {
  readonly children: ReactNode;
  readonly className?: string;
  readonly onClick?: () => void;
};

/**
 * Animated card component with consistent animations
 * 
 * Features:
 * - Fade in and slide up animation
 * - Hover scale effect
 * - Consistent styling
 */
export const AnimatedCard = ({
  children,
  className,
  onClick,
}: AnimatedCardProps): JSX.Element => {
  return (
    <motion.article
      {...cardAnimations}
      onClick={onClick}
      className={cn(
        "bg-card border rounded-lg p-6 hover:shadow-lg transition-shadow",
        className
      )}
    >
      {children}
    </motion.article>
  );
};



