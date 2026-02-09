'use client';

import React, { memo } from 'react';
import { useAuth } from '@/lib/contexts/AuthContext';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import Link from 'next/link';

interface ProtectedPageProps {
  children: React.ReactNode;
  message?: string;
  actionLabel?: string;
  actionHref?: string;
}

export const ProtectedPage: React.FC<ProtectedPageProps> = memo(({
  children,
  message = 'Sign in to access this page',
  actionLabel = 'Go home',
  actionHref = '/',
}) => {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center px-4">
        <Card className="p-8 text-center max-w-md">
          <p className="text-gray-600 dark:text-gray-400 mb-4">{message}</p>
          <Link href={actionHref}>
            <Button>{actionLabel}</Button>
          </Link>
        </Card>
      </div>
    );
  }

  return <>{children}</>;
});

ProtectedPage.displayName = 'ProtectedPage';

