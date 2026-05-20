import { useState, useCallback, useEffect } from 'react';
import { useLocalStorage } from './use-local-storage';
import { EDITOR_CONSTANTS } from '@/lib/editor/constants';
import { toast } from 'sonner';
import { useHotkeys } from 'react-hotkeys-hook';
import { TLShape } from '@tldraw/tldraw';

interface EditorState {
  title: string;
  isSaving: boolean;
  showGrid: boolean;
  lastSaved: Date | null;
  autoSave: boolean;
  zoom: number;
  selectedShapes: string[];
  history: {
    undo: () => void;
    redo: () => void;
  };
}

export function useEditorState(editor: any) {
  const [title, setTitle] = useState('Nuevo Diseño');
  const [isSaving, setIsSaving] = useState(false);
  const [showGrid, setShowGrid] = useLocalStorage('editor-show-grid', true);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [autoSave, setAutoSave] = useLocalStorage('editor-auto-save', true);
  const [zoom, setZoom] = useState(1);
  const [selectedShapes, setSelectedShapes] = useState<string[]>([]);

  // Configurar atajos de teclado
  useHotkeys(EDITOR_CONSTANTS.SHORTCUTS.SAVE.join(', '), (e) => {
    e.preventDefault();
    handleSave();
  });

  useHotkeys(EDITOR_CONSTANTS.SHORTCUTS.UNDO.join(', '), (e) => {
    e.preventDefault();
    if (editor) editor.undo();
  });

  useHotkeys(EDITOR_CONSTANTS.SHORTCUTS.REDO.join(', '), (e) => {
    e.preventDefault();
    if (editor) editor.redo();
  });

  useHotkeys(EDITOR_CONSTANTS.SHORTCUTS.TOGGLE_GRID.join(', '), (e) => {
    e.preventDefault();
    setShowGrid(!showGrid);
  });

  useHotkeys(EDITOR_CONSTANTS.SHORTCUTS.ZOOM_IN.join(', '), (e) => {
    e.preventDefault();
    handleZoomIn();
  });

  useHotkeys(EDITOR_CONSTANTS.SHORTCUTS.ZOOM_OUT.join(', '), (e) => {
    e.preventDefault();
    handleZoomOut();
  });

  useHotkeys(EDITOR_CONSTANTS.SHORTCUTS.RESET_ZOOM.join(', '), (e) => {
    e.preventDefault();
    handleResetZoom();
  });

  // Auto-guardado
  useEffect(() => {
    if (autoSave) {
      const autoSaveInterval = setInterval(() => {
        handleSave(true);
      }, EDITOR_CONSTANTS.AUTO_SAVE_INTERVAL);

      return () => clearInterval(autoSaveInterval);
    }
  }, [autoSave]);

  const handleSave = useCallback(async (isAutoSave = false) => {
    if (isAutoSave && !autoSave) return;

    setIsSaving(true);
    try {
      // Aquí iría la lógica para guardar el diseño
      await new Promise(resolve => setTimeout(resolve, 1000));
      setLastSaved(new Date());
      
      if (!isAutoSave) {
        toast.success('Diseño guardado correctamente');
      }
    } catch (error) {
      toast.error('Error al guardar el diseño');
    } finally {
      setIsSaving(false);
    }
  }, [autoSave]);

  const handleZoomIn = useCallback(() => {
    if (editor) {
      const newZoom = Math.min(zoom + EDITOR_CONSTANTS.DEFAULT_ZOOM_LEVELS.STEP, EDITOR_CONSTANTS.DEFAULT_ZOOM_LEVELS.MAX);
      editor.zoomTo(newZoom);
      setZoom(newZoom);
    }
  }, [editor, zoom]);

  const handleZoomOut = useCallback(() => {
    if (editor) {
      const newZoom = Math.max(zoom - EDITOR_CONSTANTS.DEFAULT_ZOOM_LEVELS.STEP, EDITOR_CONSTANTS.DEFAULT_ZOOM_LEVELS.MIN);
      editor.zoomTo(newZoom);
      setZoom(newZoom);
    }
  }, [editor, zoom]);

  const handleResetZoom = useCallback(() => {
    if (editor) {
      editor.zoomTo(1);
      setZoom(1);
    }
  }, [editor]);

  const handleShapeSelect = useCallback((shapeIds: string[]) => {
    setSelectedShapes(shapeIds);
  }, []);

  const handleShapeCreate = useCallback((shape: TLShape) => {
    if (editor) {
      editor.createShape(shape);
    }
  }, [editor]);

  const handleShapeDelete = useCallback((shapeIds: string[]) => {
    if (editor) {
      editor.deleteShapes(shapeIds);
    }
  }, [editor]);

  const handleShapeUpdate = useCallback((shapeId: string, updates: Partial<TLShape>) => {
    if (editor) {
      editor.updateShape(shapeId, updates);
    }
  }, [editor]);

  const handleGridToggle = useCallback(() => {
    setShowGrid(!showGrid);
  }, [showGrid]);

  const handleAutoSaveToggle = useCallback(() => {
    setAutoSave(!autoSave);
  }, [autoSave]);

  const handleUndo = useCallback(() => {
    if (editor) editor.undo();
  }, [editor]);

  const handleRedo = useCallback(() => {
    if (editor) editor.redo();
  }, [editor]);

  return {
    title,
    setTitle,
    isSaving,
    showGrid,
    setShowGrid,
    lastSaved,
    autoSave,
    setAutoSave,
    zoom,
    selectedShapes,
    handleSave,
    handleGridToggle,
    handleAutoSaveToggle,
    handleUndo,
    handleRedo,
    handleZoomIn,
    handleZoomOut,
    handleResetZoom,
    handleShapeSelect,
    handleShapeCreate,
    handleShapeDelete,
    handleShapeUpdate,
    history: {
      undo: () => editor?.undo(),
      redo: () => editor?.redo(),
    },
  };
}    