'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api-client';
import type { DocumentListItem } from '@/types/api';
import { format } from 'date-fns';
import { showToast } from '@/lib/toast';
import DocumentModal from '@/components/DocumentModal';

export default function DocumentsView() {
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);

  const loadDocuments = async () => {
    setIsLoading(true);
    try {
      const response = await apiClient.listDocuments(50, 0);
      setDocuments(response.documents);
    } catch (error: any) {
      showToast(error.message || 'Error al cargar documentos', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const handleViewDocument = (taskId: string) => {
    setSelectedTaskId(taskId);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Documentos Generados</h2>
          <p className="text-gray-600">Accede a todos tus documentos generados</p>
        </div>
        <button onClick={loadDocuments} className="btn btn-secondary" disabled={isLoading}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
            <path d="M3 21v-5h5" />
          </svg>
          Actualizar
        </button>
      </div>

      {isLoading ? (
        <div className="card text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Cargando documentos...</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="card text-center py-12">
          <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
          </svg>
          <p className="text-gray-600">No hay documentos disponibles</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {documents.map((doc) => (
            <div key={doc.task_id} className="card hover:shadow-md transition-shadow cursor-pointer" onClick={() => handleViewDocument(doc.task_id)}>
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  {doc.business_area && (
                    <span className="badge badge-info text-xs mb-2">{doc.business_area}</span>
                  )}
                  {doc.document_type && (
                    <span className="badge badge-info text-xs ml-2">{doc.document_type}</span>
                  )}
                </div>
              </div>
              <p className="text-gray-700 mb-4 line-clamp-3">{doc.query_preview}</p>
              <div className="text-sm text-gray-500">
                {format(new Date(doc.created_at), "PPp")}
              </div>
              <button className="btn btn-primary w-full mt-4 text-sm">
                Ver Documento
              </button>
            </div>
          ))}
        </div>
      )}

      {selectedTaskId && (
        <DocumentModal taskId={selectedTaskId} onClose={() => setSelectedTaskId(null)} />
      )}
    </div>
  );
}

