'use client';

import { useEffect, useState, useCallback, use } from 'react';
import { Tldraw, useEditor, TLShape, createShapeId } from '@tldraw/tldraw';
import { WebsocketProvider } from 'y-websocket';
import * as Y from 'yjs';
import { EditorToolbar } from '@/components/editor/EditorToolbar';
import { DrawingTools } from '@/components/editor/DrawingTools';
import { LayersPanel } from '@/components/editor/LayersPanel';
import { useEditorState } from '@/hooks/use-editor-state';
import { useLayers } from '@/hooks/use-layers';
import { EDITOR_CONSTANTS } from '@/lib/editor/constants';
import { Template } from '@/lib/editor/templates';
import { toast } from 'sonner';
import { useHotkeys } from 'react-hotkeys-hook';
import { useLocalStorage } from '@/hooks/use-local-storage';

interface EditorPageProps {
  params: Promise<{
    id: string;
  }>;
}

export default function EditorPage({ params }: EditorPageProps) {
  const resolvedParams = use(params);
  const [doc, setDoc] = useState<Y.Doc | null>(null);
  const [provider, setProvider] = useState<WebsocketProvider | null>(null);
  const {
    title,
    isSaving,
    showGrid,
    lastSaved,
    autoSave,
    zoom,
    selectedShapes,
    history,
    handleSave,
    handleGridToggle,
    handleAutoSaveToggle,
    handleUndo,
    handleRedo,
    handleShapeSelect,
    handleShapeCreate,
    handleShapeDelete,
    handleShapeUpdate,
  } = useEditorState(resolvedParams.id);

  const {
    layers,
    selectedLayerId,
    setSelectedLayerId,
    addLayer,
    deleteLayer,
    duplicateLayer,
    moveLayer,
    toggleLayerVisibility,
    toggleLayerLock,
    updateLayerName,
    updateLayerColor,
  } = useLayers();

  const [editor, setEditor] = useState<any>(null);

  // Configurar atajos de teclado
  useHotkeys('ctrl+s, cmd+s', (e) => {
    e.preventDefault();
    handleSave();
  });

  useHotkeys('ctrl+z, cmd+z', (e) => {
    e.preventDefault();
    if (editor) editor.undo();
  });

  useHotkeys('ctrl+shift+z, cmd+shift+z', (e) => {
    e.preventDefault();
    if (editor) editor.redo();
  });

  useHotkeys('ctrl+g, cmd+g', (e) => {
    e.preventDefault();
    handleGridToggle();
  });

  useEffect(() => {
    try {
      const yDoc = new Y.Doc();
      const wsProvider = new WebsocketProvider(
        'ws://localhost:1234',
        resolvedParams.id,
        yDoc
      );

      setDoc(yDoc);
      setProvider(wsProvider);

      // Configurar auto-guardado
      if (autoSave) {
        const autoSaveInterval = setInterval(() => {
          handleSave(true);
        }, 30000); // Guardar cada 30 segundos

        return () => {
          clearInterval(autoSaveInterval);
          wsProvider.destroy();
          yDoc.destroy();
        };
      }

      return () => {
        wsProvider.destroy();
        yDoc.destroy();
      };
    } catch (error) {
      toast.error('Error al inicializar el editor');
    }
  }, [resolvedParams.id, autoSave]);

  const handleZoomIn = useCallback(() => {
    if (editor) {
      editor.zoomIn();
    }
  }, [editor]);

  const handleZoomOut = useCallback(() => {
    if (editor) {
      editor.zoomOut();
    }
  }, [editor]);

  const applyTemplate = useCallback((template: Template) => {
    if (editor) {
      try {
        // Obtén todos los IDs de shapes actuales
        const allShapeIds = editor.getShapeIds ? editor.getShapeIds() : [];
        if (Array.isArray(allShapeIds) && allShapeIds.length > 0) {
          editor.deleteShapes(allShapeIds);
        }
        template.shapes.forEach((shape) => {
          editor.createShape(shape as unknown as TLShape);
        });
        toast.success(`Plantilla "${template.name}" aplicada correctamente`);
      } catch (error) {
        toast.error('Error al aplicar la plantilla');
      }
    }
  }, [editor]);

  return (
    <div className="flex h-screen flex-col bg-background">
      <EditorToolbar
        title={title}
        onTitleChange={(newTitle) => {
          // Implement the logic to update the title
        }}
        onSave={handleSave}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        showGrid={showGrid}
        onToggleGrid={handleGridToggle}
        onApplyTemplate={applyTemplate}
      />

      {/* Canvas */}
      <div className="flex-1 overflow-hidden relative">
        {doc && (
          <Tldraw
            persistenceKey={resolvedParams.id}
            onMount={(editor) => {
              try {
                setEditor(editor);
                editor.updateInstanceState({ isGridMode: showGrid });
              } catch (error) {
                toast.error('Error al inicializar el editor');
              }
            }}
          />
        )}
        {lastSaved && (
          <div className="absolute bottom-4 right-4 text-xs text-muted-foreground">
            Último guardado: {lastSaved.toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
}          