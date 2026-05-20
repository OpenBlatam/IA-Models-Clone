'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Loading } from '@/components/ui/Loading';
import { Alert } from '@/components/ui/Alert';
import { CheckCircle } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useRouter as useI18nRouter } from '@/i18n/routing';

export default function PricingSuccessPage() {
  const searchParams = useSearchParams();
  const router = useI18nRouter();
  const t = useTranslations('pricing');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const sessionId = searchParams.get('session_id');
    
    if (!sessionId) {
      setError(t('noSessionId'));
      setLoading(false);
      return;
    }

    const verifySession = async () => {
      try {
        const response = await fetch(`/api/stripe/verify-session?session_id=${sessionId}`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || t('verificationError'));
        }

        setLoading(false);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : t('verificationError');
        setError(errorMessage);
        setLoading(false);
      }
    };

    verifySession();
  }, [searchParams, t]);

  const handleGoToDashboard = () => {
    router.push('/dashboard');
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <Loading size="lg" text={t('verifying')} />
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="flex items-center justify-center min-h-[400px]">
          <Card className="w-full max-w-md">
            <CardContent className="p-6">
              <Alert variant="error" title={t('error')}>
                {error}
              </Alert>
            </CardContent>
          </Card>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="w-full max-w-md">
          <CardContent className="p-6 text-center">
            <div className="mb-4 flex justify-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/20">
                <CheckCircle className="h-8 w-8 text-green-600 dark:text-green-400" />
              </div>
            </div>
            <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-gray-100">
              {t('successTitle')}
            </h2>
            <p className="mb-6 text-gray-600 dark:text-gray-400">
              {t('successMessage')}
            </p>
            <Button onClick={handleGoToDashboard} variant="primary" size="lg">
              {t('goToDashboard')}
            </Button>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}



