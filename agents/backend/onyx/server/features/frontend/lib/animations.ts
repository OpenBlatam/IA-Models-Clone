import { Variants } from 'framer-motion';

export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
};

export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export const fadeInDown: Variants = {
  hidden: { opacity: 0, y: -20 },
  visible: { opacity: 1, y: 0 },
};

export const fadeInLeft: Variants = {
  hidden: { opacity: 0, x: -20 },
  visible: { opacity: 1, x: 0 },
};

export const fadeInRight: Variants = {
  hidden: { opacity: 0, x: 20 },
  visible: { opacity: 1, x: 0 },
};

export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { opacity: 1, scale: 1 },
};

export const scaleOut: Variants = {
  hidden: { opacity: 1, scale: 1 },
  visible: { opacity: 0, scale: 0.8 },
};

export const slideInLeft: Variants = {
  hidden: { x: '-100%' },
  visible: { x: 0 },
};

export const slideInRight: Variants = {
  hidden: { x: '100%' },
  visible: { x: 0 },
};

export const slideInUp: Variants = {
  hidden: { y: '100%' },
  visible: { y: 0 },
};

export const slideInDown: Variants = {
  hidden: { y: '-100%' },
  visible: { y: 0 },
};

export const rotateIn: Variants = {
  hidden: { opacity: 0, rotate: -180 },
  visible: { opacity: 1, rotate: 0 },
};

export const bounce: Variants = {
  hidden: { opacity: 0, scale: 0.3 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      type: 'spring',
      stiffness: 400,
      damping: 17,
    },
  },
};

export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

export const staggerItem: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
  },
};

export const transition = {
  default: { duration: 0.3, ease: 'easeInOut' },
  spring: { type: 'spring', stiffness: 300, damping: 30 },
  smooth: { duration: 0.5, ease: [0.4, 0, 0.2, 1] },
  bounce: { type: 'spring', stiffness: 400, damping: 17 },
};

