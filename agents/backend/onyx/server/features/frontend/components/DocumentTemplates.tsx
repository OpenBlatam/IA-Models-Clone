'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiFileText, FiPlus, FiX, FiEdit2, FiTrash2 } from 'react-icons/fi';

interface CustomTemplate {
  id: string;
  name: string;
  query: string;
  business_area: string;
  document_type: string;
  createdAt: Date;
}

export default function DocumentTemplates() {
  const [templates, setTemplates] = useState<CustomTemplate[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [editing, setEditing] = useState<CustomTemplate | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    query: '',
    business_area: '',
    document_type: '',
  });

  const loadTemplates = () => {
    const stored = localStorage.getItem('bul_custom_templates');
    if (stored) {
      setTemplates(
        JSON.parse(stored).map((t: any) => ({
          ...t,
          createdAt: new Date(t.createdAt),
        }))
      );
    }
  };

  const saveTemplate = () => {
    if (!formData.name || !formData.query) return;

    const template: CustomTemplate = {
      id: editing?.id || Date.now().toString(),
      ...formData,
      createdAt: editing?.createdAt || new Date(),
    };

    const updated = editing
      ? templates.map((t) => (t.id === editing.id ? template : t))
      : [...templates, template];

    setTemplates(updated);
    localStorage.setItem('bul_custom_templates', JSON.stringify(updated));
    setFormData({ name: '', query: '', business_area: '', document_type: '' });
    setEditing(null);
  };

  const deleteTemplate = (id: string) => {
    const updated = templates.filter((t) => t.id !== id);
    setTemplates(updated);
    localStorage.setItem('bul_custom_templates', JSON.stringify(updated));
  };

  const startEdit = (template: CustomTemplate) => {
    setEditing(template);
    setFormData({
      name: template.name,
      query: template.query,
      business_area: template.business_area,
      document_type: template.document_type,
    });
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => {
          setIsOpen(true);
          loadTemplates();
        }}
        className="btn btn-secondary"
      >
        <FiFileText size={18} className="mr-2" />
        Mis Plantillas
      </button>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={() => setIsOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              Mis Plantillas Personalizadas
            </h3>
            <button onClick={() => setIsOpen(false)} className="btn-icon">
              <FiX size={20} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">
                {editing ? 'Editar Plantilla' : 'Nueva Plantilla'}
              </h4>
              <div className="space-y-3">
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="Nombre de la plantilla"
                  className="input w-full"
                />
                <textarea
                  value={formData.query}
                  onChange={(e) => setFormData({ ...formData, query: e.target.value })}
                  placeholder="Consulta de negocio..."
                  className="textarea w-full min-h-[100px]"
                />
                <div className="grid grid-cols-2 gap-3">
                  <input
                    type="text"
                    value={formData.business_area}
                    onChange={(e) => setFormData({ ...formData, business_area: e.target.value })}
                    placeholder="Área de negocio"
                    className="input"
                  />
                  <input
                    type="text"
                    value={formData.document_type}
                    onChange={(e) => setFormData({ ...formData, document_type: e.target.value })}
                    placeholder="Tipo de documento"
                    className="input"
                  />
                </div>
                <div className="flex gap-2">
                  <button onClick={saveTemplate} className="btn btn-primary flex-1">
                    <FiPlus size={18} className="mr-2" />
                    {editing ? 'Guardar' : 'Crear'}
                  </button>
                  {editing && (
                    <button
                      onClick={() => {
                        setEditing(null);
                        setFormData({ name: '', query: '', business_area: '', document_type: '' });
                      }}
                      className="btn btn-secondary"
                    >
                      Cancelar
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {templates.length === 0 ? (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <FiFileText size={48} className="mx-auto mb-2 opacity-50" />
                  <p>No hay plantillas personalizadas</p>
                </div>
              ) : (
                templates.map((template) => (
                  <motion.div
                    key={template.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg flex items-start justify-between"
                  >
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                        {template.name}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {template.query.substring(0, 100)}...
                      </p>
                      <div className="flex gap-2 text-xs text-gray-500 dark:text-gray-500">
                        <span>{template.business_area}</span>
                        <span>•</span>
                        <span>{template.document_type}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      <button
                        onClick={() => startEdit(template)}
                        className="btn-icon text-primary-600"
                      >
                        <FiEdit2 size={16} />
                      </button>
                      <button
                        onClick={() => deleteTemplate(template.id)}
                        className="btn-icon text-red-600"
                      >
                        <FiTrash2 size={16} />
                      </button>
                    </div>
                  </motion.div>
                ))
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


