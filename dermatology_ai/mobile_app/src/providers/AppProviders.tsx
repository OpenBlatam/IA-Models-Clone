import React from 'react';
import { ThemeProvider } from '../context/ThemeContext';
import { ToastProvider } from '../context/ToastContext';

interface AppProvidersProps {
  children: React.ReactNode;
}

/**
 * Centralized providers component
 * Groups all context providers in one place for better maintainability
 */
export const AppProviders: React.FC<AppProvidersProps> = React.memo(
  ({ children }) => {
    return (
      <ThemeProvider>
        <ToastProvider>{children}</ToastProvider>
      </ThemeProvider>
    );
  }
);

AppProviders.displayName = 'AppProviders';

