'use client';

import { useSession, signOut } from 'next-auth/react';
import { useRouter } from '@/i18n/routing';

export const useAuth = () => {
  const sessionResult = useSession();
  const router = useRouter();

  const session = sessionResult?.data;
  const status = sessionResult?.status || 'loading';

  const handleSignOut = async () => {
    try {
      const locale = typeof window !== 'undefined' ? window.location.pathname.split('/')[1] || 'es' : 'es';
      await signOut({
        callbackUrl: `/${locale}/login`,
        redirect: true,
      });
    } catch (error) {
      await signOut({
        callbackUrl: '/login',
        redirect: true,
      });
    }
  };

  return {
    session,
    user: session?.user,
    isLoading: status === 'loading',
    isAuthenticated: status === 'authenticated',
    signOut: handleSignOut,
  };
};

