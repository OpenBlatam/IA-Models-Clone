'use client';

import { signIn, getSession } from 'next-auth/react';
import { useEffect, useState } from 'react';
import { useRouter } from '@/i18n/routing';
import { useLocale } from 'next-intl';
import { useTranslations } from 'next-intl';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Loading } from '@/components/ui/Loading';
import { Alert } from '@/components/ui/Alert';
import { Chrome } from 'lucide-react';

export default function LoginPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const locale = useLocale();
  const t = useTranslations('auth');

  useEffect(() => {
    const checkSession = async () => {
      const session = await getSession();
      if (session) {
        router.push(`/${locale}/dashboard`);
      }
    };
    checkSession();
  }, [router, locale]);

  const handleGoogleSignIn = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await signIn('google', {
        callbackUrl: `/${locale}/dashboard`,
        redirect: true,
      });

      if (result?.error) {
        setError(result.error);
        setLoading(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t('error'));
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleGoogleSignIn();
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">{t('title')}</CardTitle>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">{t('subtitle')}</p>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert variant="error" title={t('error')}>
              {error}
            </Alert>
          )}

          <Button
            type="button"
            variant="primary"
            size="lg"
            className="w-full"
            onClick={handleGoogleSignIn}
            onKeyDown={handleKeyDown}
            disabled={loading}
            aria-label={t('signInWithGoogle')}
            tabIndex={0}
          >
            {loading ? (
              <Loading size="sm" inline />
            ) : (
              <>
                <Chrome className="mr-2 h-5 w-5" />
                {t('signInWithGoogle')}
              </>
            )}
          </Button>

          <div className="text-center text-xs text-gray-500 dark:text-gray-400">
            {t('terms')}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

