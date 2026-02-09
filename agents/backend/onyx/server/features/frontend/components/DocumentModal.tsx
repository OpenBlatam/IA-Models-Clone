'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api-client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { showToast } from '@/lib/toast';
import { motion, AnimatePresence } from 'framer-motion';
import { FiCopy, FiDownload, FiX, FiFileText, FiPrinter, FiShare2 } from 'react-icons/fi';
import { useHotkeys } from 'react-hotkeys-hook';
import ShareModal from './ShareModal';
import { FavoriteButton } from './FavoritesManager';
import PresentationMode from './PresentationMode';
import ReadingMode from './ReadingMode';
import CommentsPanel from './CommentsPanel';
import ExportAdvanced from './ExportAdvanced';
import { exportToHTML } from '@/lib/export-utils';
import { exportToPDF } from '@/lib/pdf-export';
import { FiMaximize2, FiMessageSquare, FiBook, FiDownload, FiGitBranch } from 'react-icons/fi';
import DocumentHistory from './DocumentHistory';
import BookmarksManager from './BookmarksManager';
import FullscreenMode from './FullscreenMode';

interface DocumentModalProps {
  taskId: string;
  onClose: () => void;
}

export default function DocumentModal({ taskId, onClose }: DocumentModalProps) {
  const [document, setDocument] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCopying, setIsCopying] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [showPresentation, setShowPresentation] = useState(false);
  const [showReadingMode, setShowReadingMode] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [showExportAdvanced, setShowExportAdvanced] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [isExportingPDF, setIsExportingPDF] = useState(false);

  useEffect(() => {
    const loadDocument = async () => {
      try {
        const doc = await apiClient.getTaskDocument(taskId);
        setDocument(doc);
      } catch (error: any) {
        showToast(error.message || 'Error al cargar documento', 'error');
        onClose();
      } finally {
        setIsLoading(false);
      }
    };

    loadDocument();
  }, [taskId, onClose]);

  // Close on Escape key
  useHotkeys('escape', () => onClose(), { enabled: true });

  const handleCopy = async () => {
    if (!document?.document?.content) return;
    
    setIsCopying(true);
    try {
      await navigator.clipboard.writeText(document.document.content);
      showToast('Documento copiado al portapapeles', 'success');
    } catch (error) {
      showToast('Error al copiar documento', 'error');
    } finally {
      setIsCopying(false);
    }
  };

  const handleDownload = async (format: 'md' | 'txt' | 'html' | 'pdf' = 'md') => {
    if (!document?.document?.content) return;
    
    if (format === 'pdf') {
      setIsExportingPDF(true);
      try {
        await exportToPDF(document.document.content, `documento-${taskId}`);
        showToast('Documento descargado como PDF', 'success');
      } catch (error: any) {
        showToast(error.message || 'Error al exportar PDF', 'error');
      } finally {
        setIsExportingPDF(false);
      }
      return;
    }
    
    if (format === 'html') {
      exportToHTML(document.document.content, `documento-${taskId}`);
      showToast('Documento descargado como HTML', 'success');
      return;
    }
    
    const extension = format === 'md' ? 'md' : 'txt';
    const mimeType = format === 'md' ? 'text/markdown' : 'text/plain';
    
    const blob = new Blob([document.document.content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `documento-${taskId}.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Documento descargado como ${extension.toUpperCase()}`, 'success');
  };

  const handlePrint = () => {
    if (!document?.document?.content) return;
    
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(`
        <html>
          <head>
            <title>Documento - ${taskId}</title>
            <style>
              body { font-family: Arial, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
              pre { white-space: pre-wrap; word-wrap: break-word; }
            </style>
          </head>
          <body>
            <pre>${document.document.content}</pre>
          </body>
        </html>
      `);
      printWindow.document.close();
      printWindow.print();
    }
  };

  return (
    <AnimatePresence>
      <div
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-white rounded-xl shadow-xl max-w-5xl w-full max-h-[90vh] flex flex-col"
        >
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div className="flex items-center gap-3">
              <FiFileText size={24} className="text-primary-600" />
              <div>
                <h3 className="text-xl font-bold text-gray-900">Documento Generado</h3>
                <p className="text-sm text-gray-500 font-mono">{taskId}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="btn-icon"
              title="Cerrar (Esc)"
            >
              <FiX size={24} />
            </button>
          </div>
        
        <FullscreenMode className="flex-1 overflow-y-auto">
          <div className="p-6">
            {isLoading ? (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
                <p className="text-gray-600 dark:text-gray-400">Cargando documento...</p>
              </div>
            ) : document?.document?.content ? (
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {document.document.content}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="text-center py-12">
                <p className="text-gray-600 dark:text-gray-400">No se pudo cargar el documento</p>
              </div>
            )}
          </div>
        </FullscreenMode>
        
          <div className="flex items-center justify-between p-6 border-t border-gray-200">
            <div className="flex items-center gap-2">
              <button
                onClick={handleCopy}
                className="btn btn-secondary"
                disabled={isLoading || isCopying || !document?.document?.content}
                title="Copiar al portapapeles"
              >
                <FiCopy size={18} />
                {isCopying ? 'Copiando...' : 'Copiar'}
              </button>
              <div className="relative group">
                <button
                  onClick={() => handleDownload('md')}
                  className="btn btn-secondary"
                  disabled={isLoading || !document?.document?.content}
                  title="Descargar como Markdown"
                >
                  <FiDownload size={18} />
                  Descargar
                </button>
                <div className="absolute left-0 top-full mt-1 hidden group-hover:block">
                  <div className="bg-white border border-gray-200 rounded-lg shadow-lg py-1 min-w-[150px]">
                    <button
                      onClick={() => handleDownload('md')}
                      className="w-full text-left px-4 py-2 hover:bg-gray-50 text-sm"
                    >
                      📄 Markdown (.md)
                    </button>
                    <button
                      onClick={() => handleDownload('txt')}
                      className="w-full text-left px-4 py-2 hover:bg-gray-50 dark:hover:bg-gray-700 text-sm"
                    >
                      📝 Texto (.txt)
                    </button>
                    <button
                      onClick={() => handleDownload('html')}
                      className="w-full text-left px-4 py-2 hover:bg-gray-50 dark:hover:bg-gray-700 text-sm"
                    >
                      🌐 HTML (.html)
                    </button>
                    <button
                      onClick={() => handleDownload('pdf')}
                      className="w-full text-left px-4 py-2 hover:bg-gray-50 dark:hover:bg-gray-700 text-sm"
                      disabled={isExportingPDF}
                    >
                      📄 PDF (.pdf) {isExportingPDF && '...'}
                    </button>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setShowPresentation(true)}
                className="btn btn-secondary"
                disabled={isLoading || !document?.document?.content}
                title="Modo Presentación"
              >
                <FiMaximize2 size={18} />
                Presentación
              </button>
              <button
                onClick={() => setShowReadingMode(true)}
                className="btn btn-secondary"
                disabled={isLoading || !document?.document?.content}
                title="Modo Lectura"
              >
                <FiBook size={18} />
                Lectura
              </button>
              <button
                onClick={handlePrint}
                className="btn btn-secondary"
                disabled={isLoading || !document?.document?.content}
                title="Imprimir"
              >
                <FiPrinter size={18} />
                Imprimir
              </button>
              <button
                onClick={() => setShowShareModal(true)}
                className="btn btn-secondary"
                disabled={isLoading || !document?.document?.content}
                title="Compartir"
              >
                <FiShare2 size={18} />
                Compartir
              </button>
              <button
                onClick={() => setShowComments(!showComments)}
                className={`btn btn-secondary ${showComments ? 'bg-primary-50 dark:bg-primary-900/30' : ''}`}
                title="Comentarios"
              >
                <FiMessageSquare size={18} />
              </button>
              <button
                onClick={() => setShowHistory(true)}
                className="btn btn-secondary"
                title="Historial de Versiones"
              >
                <FiGitBranch size={18} />
                Historial
              </button>
              <BookmarksManager
                taskId={taskId}
                title={document?.document?.content?.substring(0, 50) || 'Documento'}
                query={document?.metadata?.query || ''}
              />
              <FavoriteButton
                taskId={taskId}
                title={document?.document?.content?.substring(0, 50) || 'Documento'}
                query={document?.metadata?.query || ''}
              />
            </div>
            <button
              onClick={onClose}
              className="btn btn-primary"
            >
              Cerrar
            </button>
          </div>
        </motion.div>
      </div>
      {showShareModal && (
        <ShareModal
          taskId={taskId}
          documentTitle={document?.document?.content?.substring(0, 50) || 'Documento'}
          onClose={() => setShowShareModal(false)}
        />
      )}
      {showPresentation && document?.document?.content && (
        <PresentationMode
          content={document.document.content}
          title={document?.metadata?.query || 'Documento'}
          onClose={() => setShowPresentation(false)}
        />
      )}
      {showReadingMode && document?.document?.content && (
        <ReadingMode
          content={document.document.content}
          title={document?.metadata?.query || 'Documento'}
          onClose={() => setShowReadingMode(false)}
        />
      )}
      <CommentsPanel
        taskId={taskId}
        isOpen={showComments}
        onClose={() => setShowComments(false)}
      />
      {showExportAdvanced && document?.document?.content && (
        <ExportAdvanced
          content={document.document.content}
          filename={`documento-${taskId}`}
          isOpen={showExportAdvanced}
          onClose={() => setShowExportAdvanced(false)}
        />
      )}
      <DocumentHistory
        taskId={taskId}
        isOpen={showHistory}
        onClose={() => setShowHistory(false)}
        onRestore={(version) => {
          // Handle version restore
          showToast(`Versión ${version.version} restaurada`, 'success');
        }}
      />
    </AnimatePresence>
  );
}

