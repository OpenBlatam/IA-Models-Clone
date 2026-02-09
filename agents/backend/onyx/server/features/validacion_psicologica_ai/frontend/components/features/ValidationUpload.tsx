/**
 * Validation upload component for importing data
 */

'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, FileUpload, Button, Progress } from '@/components/ui';
import { Upload, FileText } from 'lucide-react';

export interface ValidationUploadProps {
  onUpload: (file: File) => Promise<void>;
  className?: string;
}

export const ValidationUpload: React.FC<ValidationUploadProps> = ({
  onUpload,
  className,
}) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = async (file: File) => {
    setError(null);
    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Simular progreso de carga
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      await onUpload(file);

      setUploadProgress(100);
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
      }, 500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al subir el archivo');
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" aria-hidden="true" />
          Importar Validaciones
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <FileUpload
          onFileSelect={handleFileSelect}
          accept=".json,.csv"
          maxSize={5}
          label="Seleccionar archivo de validaciones"
          disabled={isUploading}
        />

        {isUploading && (
          <div className="space-y-2">
            <Progress
              value={uploadProgress}
              label="Subiendo archivo..."
              showValue
              variant="success"
              animated
            />
          </div>
        )}

        {error && (
          <div className="p-3 bg-destructive/10 border border-destructive rounded-md">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        <div className="text-xs text-muted-foreground space-y-1">
          <p>Formatos soportados: JSON, CSV</p>
          <p>Tamaño máximo: 5MB</p>
        </div>
      </CardContent>
    </Card>
  );
};



