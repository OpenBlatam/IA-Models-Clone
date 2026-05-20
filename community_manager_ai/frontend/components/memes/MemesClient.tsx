/**
 * Memes Client Component
 * Client-side wrapper for memes page with state management
 */

'use client';

import { useState, useMemo } from 'react';
import { Layout } from '@/components/layout/Layout';
import { PageHeader } from '@/components/ui/PageHeader';
import { Modal } from '@/components/ui/Modal';
import { ConfirmDialog } from '@/components/ui/ConfirmDialog';
import { MemesList } from './MemesList';
import { MemesFilters } from './MemesFilters';
import { MemeForm } from './MemeForm';
import { Button } from '@/components/ui/Button';
import { useMemes, useCreateMeme, useDeleteMeme } from '@/hooks/useMemes';
import { Meme, MemeCreate } from '@/types';
import { Plus } from 'lucide-react';
import { unique } from '@/lib/utils/array';
import type { MemeFormData } from '@/lib/zod-schemas';

/**
 * Memes client component with data fetching
 */
export const MemesClient = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [deleteConfirm, setDeleteConfirm] = useState<{ isOpen: boolean; memeId: string | null }>({
    isOpen: false,
    memeId: null,
  });

  const filters = useMemo(
    () => ({
      category: selectedCategory || undefined,
      query: searchQuery || undefined,
    }),
    [selectedCategory, searchQuery]
  );

  const { data: memes = [], isLoading } = useMemes(filters);
  const createMeme = useCreateMeme();
  const deleteMeme = useDeleteMeme();

  const categories = useMemo(() => {
    return unique(memes.map((m) => m.category).filter(Boolean) as string[]);
  }, [memes]);

  const handleSubmit = async (data: MemeFormData & { file: FileList }) => {
    try {
      const formData = new FormData();
      formData.append('file', data.file[0]);
      
      if (data.caption) formData.append('caption', data.caption);
      if (data.category) formData.append('category', data.category);
      
      if (data.tags) {
        const tagsArray = typeof data.tags === 'string'
          ? data.tags.split(',').map((t) => t.trim()).filter(Boolean)
          : data.tags;
        if (tagsArray.length > 0) {
          formData.append('tags', JSON.stringify(tagsArray));
        }
      }

      await createMeme.mutateAsync(formData);
      setIsModalOpen(false);
    } catch (error) {
      // Error is handled by the mutation hook
      console.error('Error submitting meme:', error);
    }
  };

  const handleDelete = (memeId: string) => {
    setDeleteConfirm({ isOpen: true, memeId });
  };

  const handleDeleteConfirm = async () => {
    if (!deleteConfirm.memeId) return;
    try {
      await deleteMeme.mutateAsync(deleteConfirm.memeId);
      setDeleteConfirm({ isOpen: false, memeId: null });
    } catch (error) {
      // Error is handled by the mutation hook
      console.error('Error deleting meme:', error);
    }
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const isLoadingMutation = createMeme.isPending || deleteMeme.isPending;

  return (
    <Layout>
      <div className="space-y-6">
        <PageHeader
          title="Memes"
          description="Gestiona tu biblioteca de memes"
          actions={
            <Button onClick={() => setIsModalOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Subir Meme
            </Button>
          }
        />

        <MemesFilters
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          selectedCategory={selectedCategory}
          onCategoryChange={setSelectedCategory}
          categories={categories}
        />

        <MemesList
          memes={memes}
          isLoading={isLoading}
          onDelete={handleDelete}
          onEmptyAction={() => setIsModalOpen(true)}
        />
      </div>

      <Modal
        isOpen={isModalOpen}
        onClose={closeModal}
        title="Subir Meme"
        size="md"
      >
        <MemeForm
          onSubmit={handleSubmit}
          onCancel={closeModal}
          isLoading={isLoadingMutation}
        />
      </Modal>

      <ConfirmDialog
        isOpen={deleteConfirm.isOpen}
        onClose={() => setDeleteConfirm({ isOpen: false, memeId: null })}
        onConfirm={handleDeleteConfirm}
        title="Eliminar Meme"
        message="¿Estás seguro de que deseas eliminar este meme? Esta acción no se puede deshacer."
        confirmLabel="Eliminar"
        variant="danger"
        loading={isLoadingMutation}
      />
    </Layout>
  );
};


