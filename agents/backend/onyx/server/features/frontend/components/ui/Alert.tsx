'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { FiAlertCircle, FiCheckCircle, FiInfo, FiAlertTriangle, FiX } from 'react-icons/fi';

interface AlertProps {
  variant?: 'success' | 'error' | 'warning' | 'info';
  title?: string;
  children: ReactNode;
  onClose?: () => void;
  dismissible?: boolean;
}

const variantStyles = {
  success: {
    container: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
    icon: 'text-green-600 dark:text-green-400',
    title: 'text-green-800 dark:text-green-200',
    text: 'text-green-700 dark:text-green-300',
  },
  error: {
    container: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
    icon: 'text-red-600 dark:text-red-400',
    title: 'text-red-800 dark:text-red-200',
    text: 'text-red-700 dark:text-red-300',
  },
  warning: {
    container: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
    icon: 'text-yellow-600 dark:text-yellow-400',
    title: 'text-yellow-800 dark:text-yellow-200',
    text: 'text-yellow-700 dark:text-yellow-300',
  },
  info: {
    container: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    icon: 'text-blue-600 dark:text-blue-400',
    title: 'text-blue-800 dark:text-blue-200',
    text: 'text-blue-700 dark:text-blue-300',
  },
};

const icons = {
  success: FiCheckCircle,
  error: FiAlertCircle,
  warning: FiAlertTriangle,
  info: FiInfo,
};

export function Alert({
  variant = 'info',
  title,
  children,
  onClose,
  dismissible = false,
}: AlertProps) {
  const styles = variantStyles[variant];
  const Icon = icons[variant];

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`
        border rounded-lg p-4
        ${styles.container}
        ${dismissible ? 'pr-10' : ''}
      `}
    >
      <div className="flex items-start">
        <Icon size={20} className={`${styles.icon} mt-0.5 mr-3 flex-shrink-0`} />
        <div className="flex-1">
          {title && (
            <h4 className={`font-semibold mb-1 ${styles.title}`}>{title}</h4>
          )}
          <div className={styles.text}>{children}</div>
        </div>
        {dismissible && onClose && (
          <button
            onClick={onClose}
            className={`ml-3 ${styles.icon} hover:opacity-70 transition-opacity`}
            aria-label="Cerrar"
          >
            <FiX size={18} />
          </button>
        )}
      </div>
    </motion.div>
  );
}

