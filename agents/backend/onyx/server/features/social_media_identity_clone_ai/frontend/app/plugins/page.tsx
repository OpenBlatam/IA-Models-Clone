'use client';

import { useQuery } from 'react-query';
import { useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';

interface Plugin {
  plugin_id: string;
  name: string;
  description?: string;
  version?: string;
  enabled?: boolean;
}

const PluginsPage = (): JSX.Element => {
  const router = useRouter();
  const { data: plugins, isLoading } = useQuery<Plugin[]>('plugins', () => apiClient.getPlugins());

  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Plugins</h1>
        {!plugins || plugins.length === 0 ? (
          <Card>
            <EmptyState title="No plugins" description="Plugins will appear here when available" />
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {plugins.map((plugin) => (
              <Card
                key={plugin.plugin_id}
                className="cursor-pointer hover:shadow-lg transition-shadow"
                onClick={() => router.push(`/plugins/${plugin.plugin_id}`)}
                role="button"
                tabIndex={0}
                aria-label={`View plugin ${plugin.name}`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    router.push(`/plugins/${plugin.plugin_id}`);
                  }
                }}
              >
                <div className="space-y-2">
                  <h3 className="text-xl font-semibold">{plugin.name}</h3>
                  {plugin.description && <p className="text-gray-600">{plugin.description}</p>}
                  {plugin.version && (
                    <p className="text-xs text-gray-500">Version: {plugin.version}</p>
                  )}
                  {plugin.enabled !== undefined && (
                    <span
                      className={`px-2 py-1 rounded text-xs ${
                        plugin.enabled
                          ? 'bg-green-100 text-green-700'
                          : 'bg-gray-100 text-gray-700'
                      }`}
                    >
                      {plugin.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  )}
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </PageLayout>
  );
};

export default PluginsPage;

