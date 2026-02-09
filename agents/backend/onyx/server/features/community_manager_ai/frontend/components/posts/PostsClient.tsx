/**
 * Posts Client Component
 * Client-side wrapper for posts page with state management
 */

'use client';

import { useState, useMemo } from 'react';
import { Layout } from '@/components/layout/Layout';
import { PageHeader } from '@/components/ui/PageHeader';
import { Modal } from '@/components/ui/Modal';
import { ConfirmDialog } from '@/components/ui/ConfirmDialog';
import { PostsList } from './PostsList';
import { PostsFilters } from './PostsFilters';
import { PostForm } from './PostForm';
import { Button } from '@/components/ui/Button';
import { usePosts, useCreatePost, useUpdatePost, useDeletePost, usePublishPost } from '@/hooks/usePosts';
import { Post, PostCreate } from '@/types';
import { Plus } from 'lucide-react';
import type { PostFormData } from '@/lib/zod-schemas';

/**
 * Posts client component with data fetching
 */
export const PostsClient = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingPost, setEditingPost] = useState<Post | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [deleteConfirm, setDeleteConfirm] = useState<{ isOpen: boolean; postId: string | null }>({
    isOpen: false,
    postId: null,
  });

  const { data: posts = [], isLoading } = usePosts(statusFilter === 'all' ? undefined : statusFilter);
  const createPost = useCreatePost();
  const updatePost = useUpdatePost();
  const deletePost = useDeletePost();
  const publishPost = usePublishPost();

  const filteredPosts = useMemo(() => {
    if (!posts) return [];
    return posts.filter((post) => {
      const matchesSearch = searchQuery === '' || post.content.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = statusFilter === 'all' || post.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [posts, searchQuery, statusFilter]);

  const handleSubmit = async (data: PostFormData) => {
    try {
      const postData: PostCreate = {
        ...data,
        tags: data.tags
          ? typeof data.tags === 'string'
            ? data.tags.split(',').map((t) => t.trim()).filter(Boolean)
            : data.tags
          : undefined,
      };

      if (editingPost) {
        await updatePost.mutateAsync({ postId: editingPost.post_id, post: postData });
      } else {
        await createPost.mutateAsync(postData);
      }

      setIsModalOpen(false);
      setEditingPost(null);
    } catch (error) {
      // Error is handled by the mutation hooks
      console.error('Error submitting post:', error);
    }
  };

  const handleEdit = (post: Post) => {
    setEditingPost(post);
    setIsModalOpen(true);
  };

  const handleDelete = (postId: string) => {
    setDeleteConfirm({ isOpen: true, postId });
  };

  const handleDeleteConfirm = async () => {
    if (!deleteConfirm.postId) return;
    try {
      await deletePost.mutateAsync(deleteConfirm.postId);
      setDeleteConfirm({ isOpen: false, postId: null });
    } catch (error) {
      // Error is handled by the mutation hook
      console.error('Error deleting post:', error);
    }
  };

  const handlePublish = async (postId: string) => {
    try {
      await publishPost.mutateAsync(postId);
    } catch (error) {
      // Error is handled by the mutation hook
      console.error('Error publishing post:', error);
    }
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setEditingPost(null);
  };

  const isLoadingMutation = createPost.isPending || updatePost.isPending || deletePost.isPending || publishPost.isPending;

  return (
    <Layout>
      <div className="space-y-6">
        <PageHeader
          title="Posts"
          description="Gestiona tus publicaciones en redes sociales"
          actions={
            <Button onClick={() => setIsModalOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Nuevo Post
            </Button>
          }
        />

        <PostsFilters
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          statusFilter={statusFilter}
          onStatusFilterChange={setStatusFilter}
        />

        <PostsList
          posts={filteredPosts}
          isLoading={isLoading}
          onEdit={handleEdit}
          onDelete={handleDelete}
          onPublish={handlePublish}
          onEmptyAction={() => setIsModalOpen(true)}
        />
      </div>

      <Modal
        isOpen={isModalOpen}
        onClose={closeModal}
        title={editingPost ? 'Editar Post' : 'Nuevo Post'}
        size="lg"
      >
        <PostForm
          post={editingPost}
          onSubmit={handleSubmit}
          onCancel={closeModal}
          isLoading={isLoadingMutation}
        />
      </Modal>

      <ConfirmDialog
        isOpen={deleteConfirm.isOpen}
        onClose={() => setDeleteConfirm({ isOpen: false, postId: null })}
        onConfirm={handleDeleteConfirm}
        title="Eliminar Post"
        message="¿Estás seguro de que deseas eliminar este post? Esta acción no se puede deshacer."
        confirmLabel="Eliminar"
        variant="danger"
        loading={isLoadingMutation}
      />
    </Layout>
  );
};


