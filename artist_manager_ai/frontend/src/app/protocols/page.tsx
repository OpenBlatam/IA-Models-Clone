'use client';

import { useProtocols, useDeleteProtocol } from '@/hooks/use-protocols';
import { useArtist } from '@/hooks/use-artist';
import { useDeleteConfirmation } from '@/hooks/use-delete-confirmation';
import { PageLayout, PageHeader } from '@/components/layout';
import { LoadingSpinner, EmptyState, Badge, Card, CardHeader, CardTitle, CardContent, ActionButtons } from '@/components/ui';
import { FileText } from 'lucide-react';

const ProtocolsPage = () => {
  const { artistId } = useArtist();
  const { data: protocols, isLoading } = useProtocols(artistId);
  const deleteProtocol = useDeleteProtocol(artistId);

  const handleDelete = useDeleteConfirmation<string>({
    onConfirm: async (protocolId) => {
      if (protocolId) {
        await deleteProtocol.mutateAsync(protocolId);
      }
    },
    message: '¿Estás seguro de que deseas eliminar este protocolo?',
    successMessage: 'Protocolo eliminado exitosamente',
    errorMessage: 'Error al eliminar el protocolo',
  });

  const getPriorityVariant = (priority: string) => {
    const variants: Record<string, 'default' | 'success' | 'warning' | 'danger' | 'info'> = {
      critical: 'danger',
      high: 'warning',
      medium: 'info',
      low: 'default',
    };
    return variants[priority] || 'default';
  };

  if (isLoading) {
    return <LoadingSpinner message="Cargando protocolos..." fullScreen />;
  }

  return (
    <PageLayout>
      <PageHeader title="Protocolos" actionLabel="Nuevo Protocolo" actionHref="/protocols/new" />

      {!protocols || protocols.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No hay protocolos creados"
          description="Comienza creando tu primer protocolo"
          actionLabel="Crear Protocolo"
          actionHref="/protocols/new"
        />
      ) : (
        <DataGrid columns={3} gap="lg">
          {protocols.map((protocol) => (
            <Card key={protocol.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <CardTitle className="text-lg">{protocol.title}</CardTitle>
                  <Badge variant={getPriorityVariant(protocol.priority)} size="sm">
                    {protocol.priority}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 mb-4">
                  <p className="text-sm text-gray-700 line-clamp-2">{protocol.description}</p>
                  {protocol.rules.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-gray-900 mb-1">Reglas</p>
                      <div className="flex flex-wrap gap-1">
                        {protocol.rules.slice(0, 3).map((rule, index) => (
                          <span key={index} className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                            {rule}
                          </span>
                        ))}
                        {protocol.rules.length > 3 && (
                          <span className="text-xs text-gray-400 px-2 py-1">
                            +{protocol.rules.length - 3} más
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                  {protocol.do_s.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-green-700 mb-1">Hacer</p>
                      <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                        {protocol.do_s.slice(0, 2).map((doItem, index) => (
                          <li key={index}>{doItem}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                <ActionButtons
                  viewHref={`/protocols/${protocol.id}`}
                  onDelete={() => handleDelete(protocol.id)}
                  deleteLabel="Eliminar protocolo"
                  showEdit={false}
                />
              </CardContent>
            </Card>
          ))}
        </DataGrid>
      )}
    </PageLayout>
  );
};

export default ProtocolsPage;

