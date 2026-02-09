'use client';

import { useQuery } from 'react-query';
import { useParams, useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import StatusBadge from '@/components/UI/StatusBadge';

interface Plugin {
  plugin_id: string;
  name: string;
  description?: string;
  version?: string;
  enabled?: boolean;
  config?: Record<string, unknown>;
}

const PluginDetailPage = (): JSX.Element => {
  const params = useParams();
  const router = useRouter();
  const pluginId = params.id as string;

  const { data: plugin, isLoading } = useQuery<Plugin>(
    ['plugin', pluginId],
    () => apiClient.getPlugin(pluginId),
    { enabled: !!pluginId }
  );

  const handleBack = (): void => {
    router.push('/plugins');
  };

  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  if (!plugin) {
    return (
      <PageLayout>
        <Card>
          <p className="text-center text-gray-600 py-8">Plugin not found</p>
          <div className="text-center">
            <Button onClick={handleBack}>Back to Plugins</Button>
          </div>
        </Card>
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Plugin Details</h1>
          <Button variant="secondary" onClick={handleBack}>
            Back
          </Button>
        </div>

        <Card>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-semibold">{plugin.name}</h2>
              {plugin.enabled !== undefined && (
                <StatusBadge
                  status={plugin.enabled ? 'Enabled' : 'Disabled'}
                  variant={plugin.enabled ? 'success' : 'pending'}
                />
              )}
            </div>

            {plugin.description && (
              <div>
                <p className="text-sm text-gray-600 mb-1">Description</p>
                <p className="text-gray-900">{plugin.description}</p>
              </div>
            )}

            {plugin.version && (
              <div>
                <p className="text-sm text-gray-600 mb-1">Version</p>
                <p className="text-gray-900">{plugin.version}</p>
              </div>
            )}

            {plugin.config && Object.keys(plugin.config).length > 0 && (
              <div>
                <p className="text-sm text-gray-600 mb-1">Configuration</p>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <pre className="text-sm overflow-auto">
                    {JSON.stringify(plugin.config, null, 2)}
                  </pre>
                </div>
              </div>
            )}

            <div className="pt-4 border-t">
              <p className="text-xs text-gray-500">Plugin ID</p>
              <p className="text-sm font-mono">{plugin.plugin_id}</p>
            </div>
          </div>
        </Card>
      </div>
    </PageLayout>
  );
};

export default PluginDetailPage;



