import { useState, useCallback } from 'react';
import { nanoid } from 'nanoid';

export interface Layer {
  id: string;
  name: string;
  visible: boolean;
  locked: boolean;
  type: string;
  color?: string;
}

export function useLayers(initialLayers: Layer[] = []) {
  const [layers, setLayers] = useState<Layer[]>(initialLayers);
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);

  const addLayer = useCallback((type: string = 'default') => {
    const newLayer: Layer = {
      id: nanoid(),
      name: `Capa ${layers.length + 1}`,
      visible: true,
      locked: false,
      type,
    };
    setLayers((prev) => [...prev, newLayer]);
    setSelectedLayerId(newLayer.id);
  }, [layers.length]);

  const deleteLayer = useCallback((layerId: string) => {
    setLayers((prev) => prev.filter((layer) => layer.id !== layerId));
    if (selectedLayerId === layerId) {
      setSelectedLayerId(null);
    }
  }, [selectedLayerId]);

  const duplicateLayer = useCallback((layerId: string) => {
    setLayers((prev) => {
      const layerToDuplicate = prev.find((layer) => layer.id === layerId);
      if (!layerToDuplicate) return prev;

      const newLayer: Layer = {
        ...layerToDuplicate,
        id: nanoid(),
        name: `${layerToDuplicate.name} (copia)`,
      };

      return [...prev, newLayer];
    });
  }, []);

  const moveLayer = useCallback((layerId: string, direction: 'up' | 'down') => {
    setLayers((prev) => {
      const index = prev.findIndex((layer) => layer.id === layerId);
      if (index === -1) return prev;

      const newLayers = [...prev];
      const newIndex = direction === 'up' ? index + 1 : index - 1;

      if (newIndex < 0 || newIndex >= newLayers.length) return prev;

      [newLayers[index], newLayers[newIndex]] = [newLayers[newIndex], newLayers[index]];
      return newLayers;
    });
  }, []);

  const toggleLayerVisibility = useCallback((layerId: string) => {
    setLayers((prev) =>
      prev.map((layer) =>
        layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
      )
    );
  }, []);

  const toggleLayerLock = useCallback((layerId: string) => {
    setLayers((prev) =>
      prev.map((layer) =>
        layer.id === layerId ? { ...layer, locked: !layer.locked } : layer
      )
    );
  }, []);

  const updateLayerName = useCallback((layerId: string, name: string) => {
    setLayers((prev) =>
      prev.map((layer) =>
        layer.id === layerId ? { ...layer, name } : layer
      )
    );
  }, []);

  const updateLayerColor = useCallback((layerId: string, color: string) => {
    setLayers((prev) =>
      prev.map((layer) =>
        layer.id === layerId ? { ...layer, color } : layer
      )
    );
  }, []);

  return {
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
  };
} 