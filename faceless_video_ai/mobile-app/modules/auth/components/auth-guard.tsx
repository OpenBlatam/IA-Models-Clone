import React, { useEffect } from 'react';
import { useRouter, useSegments } from 'expo-router';
import { useAuth } from '@/contexts/auth-context';
import { Loading } from '@/components/ui/loading';

interface AuthGuardProps {
  children: React.ReactNode;
  requireAuth?: boolean;
}

export function AuthGuard({ children, requireAuth = true }: AuthGuardProps) {
  const { isAuthenticated, isLoading } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    if (isLoading) return;

    const inAuthGroup = segments[0] === '(auth)';

    if (requireAuth && !isAuthenticated && !inAuthGroup) {
      router.replace('/(auth)/login');
    } else if (!requireAuth && isAuthenticated && inAuthGroup) {
      router.replace('/(tabs)');
    }
  }, [isAuthenticated, isLoading, segments, requireAuth, router]);

  if (isLoading) {
    return <Loading message="Loading..." />;
  }

  return <>{children}</>;
}

