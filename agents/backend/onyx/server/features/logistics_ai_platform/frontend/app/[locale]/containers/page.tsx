'use client';

import { useQuery } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import Navbar from '@/components/layout/navbar';
import { containersApi } from '@/lib/api/containers';
import { formatDate } from '@/lib/utils';

const ContainersPage = () => {
  const t = useTranslations('containers');
  const pathname = usePathname();
  const locale = pathname?.split('/')[1] || 'en';

  const { data: containers = [], isLoading, error } = useQuery({
    queryKey: ['containers'],
    queryFn: () => containersApi.getAll(),
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">{t('title')}</h1>
              <p className="text-muted-foreground">Manage containers</p>
            </div>
            <Link href={`/${locale}/containers/create`}>
              <Button aria-label={t('createContainer')}>{t('createContainer')}</Button>
            </Link>
          </div>
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">Loading...</p>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">{t('title')}</h1>
              <p className="text-muted-foreground">Manage containers</p>
            </div>
            <Link href={`/${locale}/containers/create`}>
              <Button aria-label={t('createContainer')}>{t('createContainer')}</Button>
            </Link>
          </div>
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-destructive">Error loading containers. Please try again later.</p>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">{t('title')}</h1>
            <p className="text-muted-foreground">Manage containers</p>
          </div>
          <Link href={`/${locale}/containers/create`}>
            <Button aria-label={t('createContainer')}>{t('createContainer')}</Button>
          </Link>
        </div>

        {containers.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">No containers found.</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {containers.map((container) => (
              <Card key={container.container_id}>
                <CardHeader>
                  <CardTitle>{container.container_number}</CardTitle>
                  <CardDescription>
                    {t('containerType')}: {container.container_type}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <p className="text-sm">
                      <span className="font-medium">{t('status')}:</span> {container.status}
                    </p>
                    {container.location && (
                      <p className="text-sm">
                        <span className="font-medium">Location:</span> {container.location.city},{' '}
                        {container.location.country}
                      </p>
                    )}
                    {container.loaded_at && (
                      <p className="text-sm">
                        <span className="font-medium">Loaded:</span> {formatDate(container.loaded_at)}
                      </p>
                    )}
                    <Link
                      href={`/${locale}/containers/${container.container_id}`}
                      className="mt-4 block"
                      aria-label={`View details for container ${container.container_id}`}
                    >
                      <Button variant="outline" className="w-full">
                        View Details
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default ContainersPage;
