'use client';

import { useState, useCallback, useRef } from 'react';
import ImageUpload from '@/components/ImageUpload';
import AnalysisOptions from '@/components/AnalysisOptions';
import AnalysisButton from '@/components/AnalysisButton';
import ResultsDisplay from '@/components/ResultsDisplay';
import StatusIndicator from '@/components/StatusIndicator';
import ErrorDisplay from '@/components/ErrorDisplay';
import { analyzeImage } from '@/lib/api';
import type { AnalysisResult, AnalysisOptions as Options } from '@/types';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [options, setOptions] = useState<Options>({
    enhance: true,
    advanced: true,
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<{ message: string; type: 'success' | 'error' | 'warning' } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setError(null);
    setResults(null);
    
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file);
    } else {
      setError('Por favor, sube solo archivos de imagen');
    }
  }, [handleFileSelect]);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) {
      setError('Por favor, selecciona una imagen primero');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    setStatus({ message: '⏳ Analizando imagen...', type: 'warning' });

    try {
      const result = await analyzeImage(selectedFile, options);
      setResults(result);
      setStatus({ message: '✅ Análisis completado exitosamente', type: 'success' });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error desconocido al analizar la imagen';
      setError(errorMessage);
      setStatus({ message: `❌ ${errorMessage}`, type: 'error' });
    } finally {
      setLoading(false);
    }
  }, [selectedFile, options]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
    setStatus(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  return (
    <div className="container">
      <header className="header">
        <h1>🧴 Dermatology AI</h1>
        <p className="subtitle">Análisis de Piel con Inteligencia Artificial</p>
      </header>

      <StatusIndicator status={status} />

      <ImageUpload
        preview={preview}
        onFileSelect={handleFileSelect}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        fileInputRef={fileInputRef}
      />

      <AnalysisOptions options={options} onChange={setOptions} />

      <div className="action-section">
        <AnalysisButton
          onClick={handleAnalyze}
          disabled={!selectedFile || loading}
          loading={loading}
        />
      </div>

      {error && <ErrorDisplay error={error} onReset={handleReset} />}

      {results && <ResultsDisplay results={results} />}
    </div>
  );
}


import { useState, useCallback, useRef } from 'react';
import ImageUpload from '@/components/ImageUpload';
import AnalysisOptions from '@/components/AnalysisOptions';
import AnalyzeButton from '@/components/AnalyzeButton';
import ResultsDisplay from '@/components/ResultsDisplay';
import ErrorDisplay from '@/components/ErrorDisplay';
import StatusIndicator from '@/components/StatusIndicator';
import { analyzeImage, checkBackendConnection } from '@/lib/api';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [enhance, setEnhance] = useState(true);
  const [advanced, setAdvanced] = useState(true);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<{ message: string; type: 'success' | 'error' | 'warning' } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check backend connection on mount
  useState(() => {
    checkBackendConnection()
      .then((connected) => {
        if (connected) {
          setStatus({ message: '✅ Backend conectado correctamente', type: 'success' });
        } else {
          setStatus({ message: '⚠️ Backend responde pero con errores', type: 'warning' });
        }
      })
      .catch(() => {
        setStatus({
          message: '❌ No se puede conectar al backend. Asegúrate que esté corriendo en el puerto 8006',
          type: 'error',
        });
      });
  });

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setError(null);
    setResults(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const analysisResults = await analyzeImage(selectedFile, {
        enhance,
        advanced,
      });
      setResults(analysisResults);
      setStatus({ message: '✅ Análisis completado exitosamente', type: 'success' });
    } catch (err: any) {
      setError(err.message || 'Error al analizar la imagen');
      setStatus({ message: '❌ Error al analizar la imagen', type: 'error' });
    } finally {
      setLoading(false);
    }
  }, [selectedFile, enhance, advanced]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  });

  return (
    <div className="container">
      <header className="header">
        <h1>🧴 Dermatology AI</h1>
        <p className="subtitle">Análisis de Piel con Inteligencia Artificial</p>
      </header>

      <StatusIndicator status={status} />

      <ImageUpload
        fileInputRef={fileInputRef}
        preview={preview}
        onFileSelect={handleFileSelect}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      />

      <AnalysisOptions enhance={enhance} advanced={advanced} onEnhanceChange={setEnhance} onAdvancedChange={setAdvanced} />

      <AnalyzeButton
        disabled={!selectedFile || loading}
        loading={loading}
        onClick={handleAnalyze}
      />

      {results && <ResultsDisplay results={results} />}

      {error && <ErrorDisplay error={error} onReset={handleReset} />}
    </div>
  );
}






