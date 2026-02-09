'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import { Platform, ContentType, type Template } from '@/types';
import { PLATFORM_OPTIONS, CONTENT_TYPE_OPTIONS } from '@/lib/constants';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import Select from '@/components/UI/Select';
import Textarea from '@/components/UI/Textarea';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import ConfirmDialog from '@/components/UI/ConfirmDialog';
import PlatformBadge from '@/components/UI/PlatformBadge';

const TemplatesPage = (): JSX.Element => {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    platform: Platform.INSTAGRAM,
    content_type: ContentType.POST,
    template: '',
  });

  const { data: templates, isLoading } = useQuery<Template[]>('templates', () =>
    apiClient.getTemplates()
  );

  const createMutation = useMutation(
    (template: Omit<Template, 'template_id' | 'created_at'>) => apiClient.createTemplate(template),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('templates');
        setShowCreateForm(false);
        resetForm();
      },
    }
  );

  const deleteMutation = useMutation((templateId: string) => apiClient.deleteTemplate(templateId), {
    onSuccess: () => {
      queryClient.invalidateQueries('templates');
    },
  });

  const handleCreate = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    createMutation.mutate(formData);
  };

  const [deleteConfirm, setDeleteConfirm] = useState<{ isOpen: boolean; templateId: string | null }>({
    isOpen: false,
    templateId: null,
  });

  const handleDelete = (templateId: string): void => {
    setDeleteConfirm({ isOpen: true, templateId });
  };

  const handleConfirmDelete = (): void => {
    if (deleteConfirm.templateId) {
      deleteMutation.mutate(deleteConfirm.templateId);
      setDeleteConfirm({ isOpen: false, templateId: null });
    }
  };

  const handleCancelDelete = (): void => {
    setDeleteConfirm({ isOpen: false, templateId: null });
  };

  const handleToggleForm = (): void => {
    setShowCreateForm(!showCreateForm);
  };

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setFormData({ ...formData, name: e.target.value });
  };

  const handlePlatformChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setFormData({ ...formData, platform: e.target.value as Platform });
  };

  const handleContentTypeChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setFormData({ ...formData, content_type: e.target.value as ContentType });
  };

  const handleTemplateChange = (e: React.ChangeEvent<HTMLTextAreaElement>): void => {
    setFormData({ ...formData, template: e.target.value });
  };

  const resetForm = (): void => {
    setFormData({
      name: '',
      platform: Platform.INSTAGRAM,
      content_type: ContentType.POST,
      template: '',
    });
  };

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
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Templates</h1>
          <Button onClick={handleToggleForm}>
            {showCreateForm ? 'Cancel' : 'Create Template'}
          </Button>
        </div>

        {showCreateForm && (
          <Card title="Create New Template" className="mb-8">
            <form onSubmit={handleCreate}>
              <div className="space-y-4">
                <Input
                  label="Template Name"
                  value={formData.name}
                  onChange={handleNameChange}
                  required
                />
                <Select
                  label="Platform"
                  value={formData.platform}
                  onChange={handlePlatformChange}
                  options={PLATFORM_OPTIONS.map((opt) => ({
                    value: opt.value as Platform,
                    label: opt.label,
                  }))}
                />
                <Select
                  label="Content Type"
                  value={formData.content_type}
                  onChange={handleContentTypeChange}
                  options={CONTENT_TYPE_OPTIONS.map((opt) => ({
                    value: opt.value as ContentType,
                    label: opt.label,
                  }))}
                />
                <Textarea
                  label="Template Content"
                  value={formData.template}
                  onChange={handleTemplateChange}
                  required
                  className="min-h-[200px]"
                />
                <Button type="submit" isLoading={createMutation.isLoading} className="w-full">
                  Create Template
                </Button>
              </div>
            </form>
          </Card>
        )}

        {!templates || templates.length === 0 ? (
          <Card>
            <EmptyState
              title="No templates found"
              description="Create your first template to get started"
              actionLabel="Create Template"
              onAction={handleToggleForm}
            />
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {templates.map((template) => (
              <Card
                key={template.template_id}
                className="cursor-pointer hover:shadow-lg transition-shadow"
                onClick={() => router.push(`/templates/${template.template_id}`)}
                role="button"
                tabIndex={0}
                aria-label={`View template ${template.name}`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    router.push(`/templates/${template.template_id}`);
                  }
                }}
              >
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xl font-semibold">{template.name}</h3>
                    <Button
                      variant="danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(template.template_id);
                      }}
                      className="text-sm"
                      aria-label={`Delete template ${template.name}`}
                    >
                      Delete
                    </Button>
                  </div>
                  <div className="flex gap-2">
                    <PlatformBadge platform={template.platform} />
                    <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                      {template.content_type}
                    </span>
                  </div>
                  <div className="bg-gray-50 p-3 rounded text-sm line-clamp-4">
                    {template.template}
                  </div>
                  <p className="text-xs text-gray-500">
                    Created: {new Date(template.created_at).toLocaleDateString()}
                  </p>
                </div>
              </Card>
            ))}
          </div>
        )}

        <ConfirmDialog
          isOpen={deleteConfirm.isOpen}
          onClose={handleCancelDelete}
          onConfirm={handleConfirmDelete}
          title="Delete Template"
          message="Are you sure you want to delete this template? This action cannot be undone."
          confirmLabel="Delete"
          cancelLabel="Cancel"
          variant="danger"
          isLoading={deleteMutation.isLoading}
        />
      </div>
    </PageLayout>
  );
};

export default TemplatesPage;

