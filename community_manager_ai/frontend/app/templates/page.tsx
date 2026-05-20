'use client';

export const dynamic = 'force-dynamic';

import { useEffect, useState } from 'react';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';
import { templatesApi } from '@/lib/api';
import { Template, TemplateCreate } from '@/types';
import { Plus, Edit, Trash2 } from 'lucide-react';
import { useForm } from 'react-hook-form';

export default function TemplatesPage() {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<Template | null>(null);
  const { register, handleSubmit, reset, formState: { errors } } = useForm<TemplateCreate>();

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    try {
      const data = await templatesApi.getAll();
      setTemplates(data);
    } catch (error) {
      console.error('Error fetching templates:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateTemplate = async (data: TemplateCreate) => {
    try {
      await templatesApi.create(data);
      setIsModalOpen(false);
      reset();
      fetchTemplates();
    } catch (error) {
      console.error('Error creating template:', error);
    }
  };

  const handleEditTemplate = async (data: TemplateCreate) => {
    if (!editingTemplate) return;
    try {
      await templatesApi.update(editingTemplate.template_id, data);
      setIsModalOpen(false);
      setEditingTemplate(null);
      reset();
      fetchTemplates();
    } catch (error) {
      console.error('Error updating template:', error);
    }
  };

  const handleDeleteTemplate = async (templateId: string) => {
    if (!confirm('¿Estás seguro de eliminar esta plantilla?')) return;
    try {
      await templatesApi.delete(templateId);
      fetchTemplates();
    } catch (error) {
      console.error('Error deleting template:', error);
    }
  };

  const openEditModal = (template: Template) => {
    setEditingTemplate(template);
    reset({
      name: template.name,
      content: template.content,
      variables: template.variables,
      category: template.category,
    });
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setEditingTemplate(null);
    reset();
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Cargando...</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Plantillas</h1>
            <p className="mt-2 text-gray-600">Gestiona tus plantillas de contenido</p>
          </div>
          <Button onClick={() => setIsModalOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Nueva Plantilla
          </Button>
        </div>

        <div className="grid gap-4">
          {templates.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-gray-500">No hay plantillas disponibles</p>
              </CardContent>
            </Card>
          ) : (
            templates.map((template) => (
              <Card key={template.template_id}>
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h3 className="text-lg font-semibold text-gray-900">{template.name}</h3>
                        {template.category && (
                          <span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded">
                            {template.category}
                          </span>
                        )}
                      </div>
                      <p className="text-gray-600 mb-3 whitespace-pre-wrap">{template.content}</p>
                      {template.variables && template.variables.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                          <span className="text-sm text-gray-500">Variables:</span>
                          {template.variables.map((variable) => (
                            <span
                              key={variable}
                              className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded"
                            >
                              {`{${variable}}`}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => openEditModal(template)}
                        aria-label="Editar plantilla"
                      >
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="danger"
                        onClick={() => handleDeleteTemplate(template.template_id)}
                        aria-label="Eliminar plantilla"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </div>

      <Modal
        isOpen={isModalOpen}
        onClose={closeModal}
        title={editingTemplate ? 'Editar Plantilla' : 'Nueva Plantilla'}
        size="lg"
      >
        <form
          onSubmit={handleSubmit(editingTemplate ? handleEditTemplate : handleCreateTemplate)}
          className="space-y-4"
        >
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
              Nombre
            </label>
            <input
              id="name"
              type="text"
              {...register('name', { required: 'El nombre es requerido' })}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            {errors.name && (
              <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>
            )}
          </div>

          <div>
            <label htmlFor="content" className="block text-sm font-medium text-gray-700 mb-1">
              Contenido (usa {'{variable}'} para variables)
            </label>
            <textarea
              id="content"
              {...register('content', { required: 'El contenido es requerido' })}
              rows={6}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="Ejemplo: ¡Hola {nombre}! Bienvenido a {empresa}."
            />
            {errors.content && (
              <p className="mt-1 text-sm text-red-600">{errors.content.message}</p>
            )}
          </div>

          <div>
            <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-1">
              Categoría (opcional)
            </label>
            <input
              id="category"
              type="text"
              {...register('category')}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>

          <div className="rounded-lg bg-blue-50 p-4">
            <p className="text-sm text-blue-800">
              <strong>Tip:</strong> Las variables en el contenido deben estar entre llaves, por
              ejemplo: {'{nombre}'}, {'{fecha}'}, etc.
            </p>
          </div>

          <div className="flex justify-end gap-2 pt-4">
            <Button type="button" variant="secondary" onClick={closeModal}>
              Cancelar
            </Button>
            <Button type="submit" variant="primary">
              {editingTemplate ? 'Guardar' : 'Crear'}
            </Button>
          </div>
        </form>
      </Modal>
    </Layout>
  );
}

