'use client';

import { useState } from 'react';
import { useLocalStorage } from '@/lib/hooks/useLocalStorage';
import { File, Plus, Trash2, Download, Upload, Copy } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Template {
  id: string;
  name: string;
  description: string;
  category: 'movement' | 'optimization' | 'monitoring' | 'custom';
  config: any;
  createdAt: Date;
}

export default function TemplatesManager() {
  const [templates, setTemplates] = useLocalStorage<Template[]>('templates', [
    {
      id: '1',
      name: 'Movimiento Circular',
      description: 'Configuración para movimiento circular suave',
      category: 'movement',
      config: { type: 'circular', radius: 0.5, speed: 0.1 },
      createdAt: new Date(),
    },
    {
      id: '2',
      name: 'Optimización Rápida',
      description: 'Configuración para optimización rápida de trayectorias',
      category: 'optimization',
      config: { algorithm: 'astar', timeout: 5 },
      createdAt: new Date(),
    },
  ]);
  const [newTemplate, setNewTemplate] = useState({ name: '', description: '', category: 'custom' as const, config: {} });

  const handleCreateTemplate = () => {
    if (!newTemplate.name.trim()) {
      toast.error('El nombre es requerido');
      return;
    }
    const template: Template = {
      id: Date.now().toString(),
      ...newTemplate,
      createdAt: new Date(),
    };
    setTemplates([...templates, template]);
    setNewTemplate({ name: '', description: '', category: 'custom', config: {} });
    toast.success('Plantilla creada');
  };

  const handleDeleteTemplate = (id: string) => {
    setTemplates(templates.filter((t) => t.id !== id));
    toast.success('Plantilla eliminada');
  };

  const handleExportTemplate = (template: Template) => {
    const dataStr = JSON.stringify(template, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${template.name}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Plantilla exportada');
  };

  const handleImportTemplate = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const template = JSON.parse(event.target?.result as string);
            setTemplates([...templates, { ...template, id: Date.now().toString() }]);
            toast.success('Plantilla importada');
          } catch (error) {
            toast.error('Error al importar plantilla');
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'movement':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'optimization':
        return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'monitoring':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      default:
        return 'bg-purple-500/20 text-purple-400 border-purple-500/50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <File className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Gestión de Plantillas</h3>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleImportTemplate}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Upload className="w-4 h-4" />
              Importar
            </button>
          </div>
        </div>

        {/* Create Template */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-semibold text-white mb-3">Crear Nueva Plantilla</h4>
          <div className="space-y-3">
            <input
              type="text"
              value={newTemplate.name}
              onChange={(e) => setNewTemplate({ ...newTemplate, name: e.target.value })}
              placeholder="Nombre de la plantilla"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <textarea
              value={newTemplate.description}
              onChange={(e) => setNewTemplate({ ...newTemplate, description: e.target.value })}
              placeholder="Descripción"
              rows={2}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
            />
            <select
              value={newTemplate.category}
              onChange={(e) => setNewTemplate({ ...newTemplate, category: e.target.value as any })}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="movement">Movimiento</option>
              <option value="optimization">Optimización</option>
              <option value="monitoring">Monitoreo</option>
              <option value="custom">Personalizada</option>
            </select>
            <button
              onClick={handleCreateTemplate}
              className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Crear Plantilla
            </button>
          </div>
        </div>

        {/* Templates List */}
        <div className="space-y-3">
          {templates.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <File className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No hay plantillas disponibles</p>
            </div>
          ) : (
            templates.map((template) => (
              <div
                key={template.id}
                className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-semibold text-white">{template.name}</h4>
                      <span className={`px-2 py-1 rounded text-xs border ${getCategoryColor(template.category)}`}>
                        {template.category}
                      </span>
                    </div>
                    <p className="text-sm text-gray-300 mb-2">{template.description}</p>
                    <p className="text-xs text-gray-400">
                      Creada: {template.createdAt.toLocaleDateString('es-ES')}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleExportTemplate(template)}
                      className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                      title="Exportar"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(JSON.stringify(template.config, null, 2));
                        toast.success('Configuración copiada');
                      }}
                      className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                      title="Copiar"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeleteTemplate(template.id)}
                      className="p-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                      title="Eliminar"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}


