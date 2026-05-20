"use client";

import { motion, AnimatePresence } from "framer-motion";
import { ReactNode } from "react";

interface MotionWrapperProps {
  children: ReactNode;
  className?: string;
  initial?: any;
  animate?: any;
  exit?: any;
  transition?: any;
  layout?: boolean;
  layoutId?: string;
}

export function MotionDiv({ children, ...props }: MotionWrapperProps) {
  return <motion.div {...props}>{children}</motion.div>;
}

export function MotionSpan({ children, ...props }: MotionWrapperProps) {
  return <motion.span {...props}>{children}</motion.span>;
}

export function MotionButton({ children, ...props }: MotionWrapperProps) {
  return <motion.button {...props}>{children}</motion.button>;
}

export { motion, AnimatePresence };
