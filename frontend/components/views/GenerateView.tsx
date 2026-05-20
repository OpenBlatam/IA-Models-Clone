'use client';

import { useState, useEffect, useRef } from 'react';
import { apiClient } from '@/lib/api-client';
import { useAppStore } from '@/store/app-store';
import type { DocumentRequest, TaskStatus, WebSocketMessage } from '@/types/api';
import { showToast } from '@/lib/toast';
import { createTaskNotification } from '@/lib/notifications';
import ProgressCard from '@/components/ProgressCard';
import TemplatesPanel from '@/components/TemplatesPanel';
import type { Template } from '@/components/TemplatesPanel';
import DocumentTemplates from '@/components/DocumentTemplates';
import WorkflowBuilder from '@/components/WorkflowBuilder';
import AutoComplete from '@/components/AutoComplete';
import LivePreview from '@/components/LivePreview';
import DocumentVersions from '@/components/DocumentVersions';
import RichTextEditor from '@/components/RichTextEditor';
import TagsManager from '@/components/TagsManager';
import DragDropZone from '@/components/DragDropZone';
import AutoSave from '@/components/AutoSave';
import VoiceInput from '@/components/VoiceInput';
import { useSearchHistory } from '@/components/SearchHistory';
import { motion } from 'framer-motion';
import { useHotkeys } from 'react-hotkeys-hook';
import { FiEye, FiFileText, FiMic } from 'react-icons/fi';
import { showToast } from '@/lib/toast';

export default function GenerateView() {
  const { currentTask, currentTaskId, setCurrentTask, setCurrentTaskId, setActiveView } = useAppStore();
  const [formData, setFormData] = useState<DocumentRequest>({
    query: '',
    business_area: '',
    document_type: '',
    priority: 3,
  });
  const [charCount, setCharCount] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [useRichEditor, setUseRichEditor] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [showVoiceInput, setShowVoiceInput] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);
  const { addToHistory } = useSearchHistory();

  useEffect(() => {
    setCharCount(formData.query.length);
  }, [formData.query]);

  // Keyboard shortcuts
  useHotkeys('ctrl+enter,cmd+enter', (e) => {
    e.preventDefault();
    if (formRef.current && !isSubmitting && formData.query.length >= 10) {
      formRef.current.requestSubmit();
    }
  }, [formData.query, isSubmitting]);

  useHotkeys('ctrl+p,cmd+p', (e) => {
    e.preventDefault();
    setShowPreview(!showPreview);
  }, [showPreview]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (formData.query.length < 10) {
      showToast('La consulta debe tener al menos 10 caracteres', 'error');
      return;
    }

    setIsSubmitting(true);
    
    // Add to search history
    addToHistory(formData.query);
    
    // Add tags to metadata
    const requestWithTags = {
      ...formData,
      metadata: {
        ...formData.metadata,
        tags,
      },
    };
    
    try {
      const response = await apiClient.generateDocument(requestWithTags);
      setCurrentTaskId(response.task_id);
      setActiveTaskId(response.task_id);
      
      // Add activity
      window.dispatchEvent(new CustomEvent('bul_activity', {
        detail: {
          id: Date.now().toString(),
          type: 'task_created',
          message: `Nueva tarea creada: ${formData.query.substring(0, 50)}...`,
          timestamp: new Date(),
          taskId: response.task_id,
        },
      }));
      
      // Initial task status
      const initialTask: TaskStatus = {
        task_id: response.task_id,
        status: 'queued',
        progress: 0,
        created_at: response.created_at,
        updated_at: response.created_at,
      };
      setCurrentTask(initialTask);
      
      // Connect WebSocket for real-time updates
      await apiClient.connectTaskWebSocket(
        response.task_id,
        (message: WebSocketMessage) => {
          if (message.type === 'task_update' && message.data) {
            const taskStatus: TaskStatus = {
              task_id: response.task_id,
              status: message.data.status,
              progress: message.data.progress || 0,
              result: message.data.result,
              error: message.data.error,
              created_at: message.timestamp,
              updated_at: message.timestamp,
              processing_time: message.data.processing_time,
            };
            setCurrentTask(taskStatus);
            
            if (message.data.status === 'completed') {
              showToast('Documento generado exitosamente', 'success');
              createTaskNotification(response.task_id, 'completed');
              setIsSubmitting(false);
              setActiveTaskId(null);
              apiClient.disconnectTaskWebSocket(response.task_id);
            } else if (message.data.status === 'failed') {
              showToast(`Error: ${message.data.error || 'Falló la generación'}`, 'error');
              createTaskNotification(response.task_id, 'failed');
              setIsSubmitting(false);
              setActiveTaskId(null);
              apiClient.disconnectTaskWebSocket(response.task_id);
            }
          } else if (message.type === 'initial_state' && message.data) {
            const taskStatus: TaskStatus = {
              task_id: response.task_id,
              status: message.data.status,
              progress: message.data.progress || 0,
              result: message.data.result,
              error: message.data.error,
              created_at: message.timestamp,
              updated_at: message.timestamp,
              processing_time: message.data.processing_time,
            };
            setCurrentTask(taskStatus);
          }
        },
        (error) => {
          console.error('WebSocket error:', error);
          showToast('Error en conexión WebSocket, usando polling', 'warning');
        }
      );
      
      showToast('Generación iniciada', 'success');
    } catch (error: any) {
      showToast(error.message || 'Error al generar documento', 'error');
      setIsSubmitting(false);
      setActiveTaskId(null);
    }
  };

  const handleCancel = async () => {
    if (!activeTaskId) return;
    
    try {
      await apiClient.cancelTask(activeTaskId);
      showToast('Tarea cancelada', 'success');
      setIsSubmitting(false);
      setActiveTaskId(null);
      apiClient.disconnectTaskWebSocket(activeTaskId);
    } catch (error: any) {
      showToast(error.message || 'Error al cancelar tarea', 'error');
    }
  };

  const handleTemplateSelect = (template: Template) => {
    setFormData({
      query: template.query,
      business_area: template.business_area,
      document_type: template.document_type,
      priority: 3,
    });
    showToast(`Plantilla "${template.name}" cargada`, 'success');
  };

  const handleFileSelect = async (file: File) => {
    try {
      const text = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.onerror = reject;
        reader.readAsText(file);
      });
      
      setFormData({ ...formData, query: text });
      setCharCount(text.length);
      setUploadedFile(file);
      showToast('Archivo cargado exitosamente', 'success');
    } catch (error) {
      showToast('Error al leer el archivo', 'error');
    }
  };

  return (
    <div className="max-w-4xl">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Generar Nuevo Documento</h2>
        <p className="text-gray-600">Crea documentos personalizados usando inteligencia artificial</p>
        <p className="text-sm text-gray-500 mt-2">
          💡 Tip: Presiona <kbd className="px-2 py-1 bg-gray-100 rounded text-xs">Ctrl+Enter</kbd> para generar rápidamente
        </p>
      </motion.div>

      {activeTaskId && currentTask && (
        <div className="mb-6">
          <ProgressCard task={currentTask} onCancel={handleCancel} />
        </div>
      )}

      <div className="flex gap-2 mb-4 flex-wrap">
        <TemplatesPanel onSelectTemplate={handleTemplateSelect} />
        <DocumentTemplates />
        <WorkflowBuilder />
      </div>

      <motion.form
        ref={formRef}
        onSubmit={handleSubmit}
        className="card space-y-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <div className="flex items-center justify-between mb-2">
            <label htmlFor="query" className="block text-sm font-medium text-gray-700 dark:text-gray-300" data-help="generate-query">
              Consulta de Negocio *
            </label>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowVoiceInput(true)}
                className="btn btn-secondary text-sm"
                title="Entrada por voz"
              >
                <FiMic size={16} />
                Voz
              </button>
              <button
                type="button"
                onClick={() => setUseRichEditor(!useRichEditor)}
                className={`btn btn-secondary text-sm ${useRichEditor ? 'bg-primary-50 dark:bg-primary-900/30' : ''}`}
                title="Editor enriquecido"
              >
                <FiFileText size={16} />
                {useRichEditor ? 'Simple' : 'Enriquecido'}
              </button>
              <button
                type="button"
                onClick={() => setShowPreview(!showPreview)}
                className="btn btn-secondary text-sm"
                title="Vista previa (Ctrl+P)"
              >
                <FiEye size={16} />
                {showPreview ? 'Ocultar' : 'Vista Previa'}
              </button>
            </div>
          </div>
          
          {useRichEditor ? (
            <RichTextEditor
              value={formData.query}
              onChange={(value) => {
                setFormData({ ...formData, query: value });
                setCharCount(value.length);
              }}
              placeholder="Describe el documento que deseas generar..."
            />
          ) : (
            <AutoComplete
              value={formData.query}
              onChange={(value) => {
                setFormData({ ...formData, query: value });
                setCharCount(value.length);
              }}
              onSelect={(value) => {
                setFormData({ ...formData, query: value });
                setCharCount(value.length);
              }}
              placeholder="Describe el documento que deseas generar. Ejemplo: 'Necesito un plan de marketing para lanzar un nuevo producto tecnológico en el mercado latinoamericano...'"
              className="mb-2"
            />
          )}
          
          <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            {charCount} / 5000 caracteres
          </div>
        </div>

        <DragDropZone
          onFileSelect={handleFileSelect}
          accept=".txt,.md,.doc,.docx"
          maxSize={5}
          className="mb-4"
        />

        <TagsManager tags={tags} onChange={setTags} />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label htmlFor="businessArea" className="block text-sm font-medium text-gray-700 mb-2">
              Área de Negocio
            </label>
            <select
              id="businessArea"
              value={formData.business_area}
              onChange={(e) => setFormData({ ...formData, business_area: e.target.value || undefined })}
              className="select"
              disabled={isSubmitting}
            >
              <option value="">Seleccionar...</option>
              <option value="marketing">Marketing</option>
              <option value="ventas">Ventas</option>
              <option value="recursos-humanos">Recursos Humanos</option>
              <option value="tecnologia">Tecnología</option>
              <option value="finanzas">Finanzas</option>
              <option value="operaciones">Operaciones</option>
              <option value="general">General</option>
            </select>
          </div>

          <div>
            <label htmlFor="documentType" className="block text-sm font-medium text-gray-700 mb-2">
              Tipo de Documento
            </label>
            <select
              id="documentType"
              value={formData.document_type}
              onChange={(e) => setFormData({ ...formData, document_type: e.target.value || undefined })}
              className="select"
              disabled={isSubmitting}
            >
              <option value="">Seleccionar...</option>
              <option value="plan">Plan</option>
              <option value="reporte">Reporte</option>
              <option value="propuesta">Propuesta</option>
              <option value="analisis">Análisis</option>
              <option value="estrategia">Estrategia</option>
              <option value="documento">Documento Estándar</option>
            </select>
          </div>

          <div>
            <label htmlFor="priority" className="block text-sm font-medium text-gray-700 mb-2">
              Prioridad
            </label>
            <select
              id="priority"
              value={formData.priority}
              onChange={(e) => setFormData({ ...formData, priority: parseInt(e.target.value) })}
              className="select"
              disabled={isSubmitting}
            >
              <option value="1">Baja (1)</option>
              <option value="2">Media-Baja (2)</option>
              <option value="3">Media (3)</option>
              <option value="4">Media-Alta (4)</option>
              <option value="5">Alta (5)</option>
            </select>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isSubmitting || formData.query.length < 10}
          >
            {isSubmitting ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Generando...
              </>
            ) : (
              <>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14" />
                  <path d="M12 5l7 7-7 7" />
                </svg>
                Generar Documento
              </>
            )}
          </button>
        </div>
      </motion.form>

      <LivePreview
        content={formData.query}
        isVisible={showPreview}
        onClose={() => setShowPreview(false)}
      />

      {formData.query.length > 20 && (
        <DocumentVersions baseQuery={formData.query} />
      )}

      <AutoSave
        data={formData}
        onSave={(saved) => {
          if (saved.query) {
            setFormData(saved);
            setCharCount(saved.query.length);
          }
        }}
        storageKey="bul_generate_autosave"
      />

      {showVoiceInput && (
        <VoiceInput
          onTranscript={(text) => {
            setFormData({ ...formData, query: text });
            setCharCount(text.length);
            setShowVoiceInput(false);
          }}
          onClose={() => setShowVoiceInput(false)}
        />
      )}
    </div>
  );
}

