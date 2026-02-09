/**
 * Platform insights component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { useConnections } from '@/hooks/useConnections';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const PlatformInsights: React.FC = () => {
  const { data: validations } = useValidations();
  const { data: connections } = useConnections();

  const platformData = React.useMemo(() => {
    if (!validations) {
      return [];
    }

    const platformCounts: Record<string, number> = {};
    validations.forEach((validation) => {
      validation.connected_platforms.forEach((platform) => {
        platformCounts[platform] = (platformCounts[platform] || 0) + 1;
      });
    });

    return Object.entries(platformCounts)
      .map(([platform, count]) => ({ platform, count }))
      .sort((a, b) => b.count - a.count);
  }, [validations]);

  if (!platformData || platformData.length === 0) {
    return (
      <Card>
        <CardContent className="py-12">
          <p className="text-center text-muted-foreground">No hay datos de plataformas</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Insights por Plataforma</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={platformData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="platform" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="hsl(var(--primary))" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};



