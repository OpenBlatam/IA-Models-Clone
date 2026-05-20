'use client';

export const dynamic = 'force-dynamic';

import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { ThemeSelect } from '@/components/ui/ThemeToggle';
import { useTranslations } from 'next-intl';

export default function SettingsPage() {
  const t = useTranslations('settings');

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">{t('title')}</h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">{t('subtitle')}</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{t('general')}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <ThemeSelect />
            <p className="text-sm text-gray-600 dark:text-gray-400">{t('comingSoon')}</p>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}

