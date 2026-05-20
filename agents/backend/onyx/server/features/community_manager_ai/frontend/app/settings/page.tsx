'use client';

export const dynamic = 'force-dynamic';

import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';

export default function SettingsPage() {
  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Configuración</h1>
          <p className="mt-2 text-gray-600">Ajustes y preferencias del sistema</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Configuración General</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600">Las opciones de configuración estarán disponibles próximamente.</p>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}

