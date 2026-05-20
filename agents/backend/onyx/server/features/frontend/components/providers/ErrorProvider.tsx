'use client';

import { createContext, useContext, ReactNode } from 'react';
import { getErrorMessage } from '@/lib/errors';
import { showToast } from '@/lib/toast';

interface ErrorContextType {
  handleError: (error: unknown, message?: string) => void;
}

const ErrorContext = createContext<ErrorContextType | undefined>(undefined);

export function ErrorProvider({ children }: { children: ReactNode }) {
  const handleError = (error: unknown, message?: string) => {
    const errorMessage = message || getErrorMessage(error);
    showToast(errorMessage, 'error');
    console.error('Error:', error);
  };

  return (
    <ErrorContext.Provider value={{ handleError }}>
      {children}
    </ErrorContext.Provider>
  );
}

export function useError() {
  const context = useContext(ErrorContext);
  if (!context) {
    throw new Error('useError must be used within ErrorProvider');
  }
  return context;
}

