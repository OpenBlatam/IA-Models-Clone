'use client';

import { useRecentManuals } from '@/lib/hooks/use-manuals';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { ManualList } from './manual/manual-list';
import Link from 'next/link';

export const RecentManuals = (): JSX.Element => {
  const { data: manuals, isLoading, error } = useRecentManuals(5);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Manuales Recientes</CardTitle>
        <CardDescription>
          {manuals && manuals.length > 0
            ? 'Los últimos manuales generados'
            : 'No hay manuales recientes'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ManualList
          manuals={manuals}
          isLoading={isLoading}
          error={error}
          emptyMessage="No hay manuales recientes"
        />
        {manuals && manuals.length > 0 && (
          <div className="mt-4">
            <Link href="/history">
              <Button variant="outline" className="w-full" aria-label="Ver todos los manuales">
                Ver Todos los Manuales
              </Button>
            </Link>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

