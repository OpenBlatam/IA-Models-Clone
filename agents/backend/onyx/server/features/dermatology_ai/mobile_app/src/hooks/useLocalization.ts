import { useState, useCallback } from 'react';
import { useStorage } from './useStorage';

type Language = 'es' | 'en';

interface Translations {
  [key: string]: {
    [lang in Language]: string;
  };
}

const translations: Translations = {
  // Common
  'common.save': { es: 'Guardar', en: 'Save' },
  'common.cancel': { es: 'Cancelar', en: 'Cancel' },
  'common.delete': { es: 'Eliminar', en: 'Delete' },
  'common.edit': { es: 'Editar', en: 'Edit' },
  'common.share': { es: 'Compartir', en: 'Share' },
  
  // Analysis
  'analysis.title': { es: 'Análisis de Piel', en: 'Skin Analysis' },
  'analysis.loading': { es: 'Analizando...', en: 'Analyzing...' },
  'analysis.complete': { es: 'Análisis Completado', en: 'Analysis Complete' },
  
  // Recommendations
  'recommendations.title': { es: 'Recomendaciones', en: 'Recommendations' },
  'recommendations.loading': { es: 'Generando recomendaciones...', en: 'Generating recommendations...' },
  
  // History
  'history.title': { es: 'Historial', en: 'History' },
  'history.empty': { es: 'No hay análisis aún', en: 'No analyses yet' },
  
  // Camera
  'camera.permission': { es: 'Permiso de Cámara', en: 'Camera Permission' },
  'camera.takePhoto': { es: 'Tomar Foto', en: 'Take Photo' },
};

export const useLocalization = () => {
  const [language, setLanguage, , isLoading] = useStorage<Language>('language', 'es');

  const t = useCallback(
    (key: string): string => {
      const translation = translations[key];
      if (!translation) return key;
      return translation[language] || translation.es;
    },
    [language]
  );

  const changeLanguage = useCallback(
    (lang: Language) => {
      setLanguage(lang);
    },
    [setLanguage]
  );

  return {
    language,
    t,
    changeLanguage,
    isLoading,
  };
};

