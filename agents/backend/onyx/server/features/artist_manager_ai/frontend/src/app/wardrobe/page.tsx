'use client';

import { useWardrobeItems, useOutfits, useDeleteWardrobeItem } from '@/hooks/use-wardrobe';
import { useArtist } from '@/hooks/use-artist';
import { useDeleteConfirmation } from '@/hooks/use-delete-confirmation';
import { getDressCodeLabel } from '@/lib/utils';
import { PageLayout, PageHeader } from '@/components/layout';
import { LoadingSpinner, EmptyState, Section, DataGrid, Card, CardHeader, CardTitle, CardContent, ActionButtons, Button, Badge, Chip } from '@/components/ui';
import { Shirt, Plus } from 'lucide-react';
import Link from 'next/link';

const WardrobePage = () => {
  const { artistId } = useArtist();
  const { data: items, isLoading: itemsLoading } = useWardrobeItems(artistId);
  const { data: outfits, isLoading: outfitsLoading } = useOutfits(artistId);
  const deleteItem = useDeleteWardrobeItem(artistId);

  const handleDelete = useDeleteConfirmation<string>({
    onConfirm: async (itemId) => {
      if (itemId) {
        await deleteItem.mutateAsync(itemId);
      }
    },
    message: '¿Estás seguro de que deseas eliminar este item?',
    successMessage: 'Item eliminado exitosamente',
    errorMessage: 'Error al eliminar el item',
  });

  if (itemsLoading || outfitsLoading) {
    return <LoadingSpinner message="Cargando guardarropa..." fullScreen />;
  }

  return (
    <PageLayout>
      <PageHeader title="Guardarropa">
        <Link href="/wardrobe/items/new">
          <Button variant="primary">
            <Plus className="w-4 h-4 mr-2" />
            Nuevo Item
          </Button>
        </Link>
        <Link href="/wardrobe/outfits/new">
          <Button variant="secondary">
            <Plus className="w-4 h-4 mr-2" />
            Nuevo Outfit
          </Button>
        </Link>
      </PageHeader>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Section title="Items">
          {!items || items.length === 0 ? (
            <EmptyState
              icon={Shirt}
              title="No hay items en el guardarropa"
              description="Comienza agregando tu primer item"
              actionLabel="Agregar Item"
              actionHref="/wardrobe/items/new"
            />
          ) : (
            <DataGrid columns={2} gap="md">
              {items.map((item) => (
                <Card key={item.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <CardTitle className="text-lg">{item.name}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3 mb-4">
                      <div className="text-sm">
                        <p className="font-medium text-gray-700 mb-1">Categoría</p>
                        <p className="text-gray-600">{item.category}</p>
                      </div>
                      <div className="text-sm">
                        <p className="font-medium text-gray-700 mb-1">Color</p>
                        <p className="text-gray-600">{item.color}</p>
                      </div>
                      {item.brand && (
                        <div className="text-sm">
                          <p className="font-medium text-gray-700 mb-1">Marca</p>
                          <p className="text-gray-600">{item.brand}</p>
                        </div>
                      )}
                      {item.dress_codes.length > 0 && (
                        <div className="text-sm">
                          <p className="font-medium text-gray-700 mb-2">Códigos de Vestimenta</p>
                          <div className="flex flex-wrap gap-1">
                            {item.dress_codes.map((dc) => (
                              <Chip key={dc} label={getDressCodeLabel(dc)} variant="primary" size="sm" />
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    <ActionButtons
                      onDelete={() => handleDelete(item.id)}
                      deleteLabel="Eliminar item"
                      showView={false}
                      showEdit={false}
                    />
                  </CardContent>
                </Card>
              ))}
            </DataGrid>
          )}
        </Section>

        <Section title="Outfits">
          {!outfits || outfits.length === 0 ? (
            <EmptyState
              icon={Shirt}
              title="No hay outfits creados"
              description="Comienza creando tu primer outfit"
              actionLabel="Crear Outfit"
              actionHref="/wardrobe/outfits/new"
            />
          ) : (
            <div className="space-y-4">
              {outfits.map((outfit) => (
                <Card key={outfit.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <CardTitle className="text-lg">{outfit.name}</CardTitle>
                      <Badge variant="info" size="sm">
                        {getDressCodeLabel(outfit.dress_code)}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="text-sm">
                        <p className="font-medium text-gray-700 mb-1">Ocasió</p>
                        <p className="text-gray-600">{outfit.occasion}</p>
                      </div>
                      <div className="text-sm">
                        <p className="font-medium text-gray-700 mb-1">Items</p>
                        <p className="text-gray-600">{outfit.items.length} items</p>
                      </div>
                      {outfit.notes && (
                        <div className="text-sm">
                          <p className="font-medium text-gray-700 mb-1">Notas</p>
                          <p className="text-gray-600">{outfit.notes}</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </Section>
      </div>
    </PageLayout>
  );
};

export default WardrobePage;

