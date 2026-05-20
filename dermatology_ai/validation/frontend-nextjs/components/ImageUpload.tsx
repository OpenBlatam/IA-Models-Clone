'use client';

import { useRef, useCallback } from 'react';

interface ImageUploadProps {
  fileInputRef: React.RefObject<HTMLInputElement>;
  preview: string | null;
  onFileSelect: (file: File) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
}

export default function ImageUpload({
  fileInputRef,
  preview,
  onFileSelect,
  onDragOver,
  onDrop,
}: ImageUploadProps) {
  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, [fileInputRef]);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file && file.type.startsWith('image/')) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  return (
    <section className="upload-section">
      <div
        className={`upload-box ${preview ? '' : 'cursor-pointer'}`}
        onClick={handleClick}
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
        {preview ? (
          <img src={preview} alt="Preview" className="preview" />
        ) : (
          <div className="upload-content">
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <h3>Sube una foto de tu piel</h3>
            <p>Haz clic aquí o arrastra una imagen</p>
            <p className="hint">Formatos: JPG, PNG (máx. 10MB)</p>
          </div>
        )}
      </div>
    </section>
  );
}


import { useRef } from 'react';

interface ImageUploadProps {
  preview: string | null;
  onFileSelect: (file: File) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
}

export default function ImageUpload({
  preview,
  onFileSelect,
  onDragOver,
  onDrop,
  fileInputRef,
}: ImageUploadProps) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 10 * 1024 * 1024) {
        alert('El archivo es demasiado grande. Máximo 10MB');
        return;
      }
      onFileSelect(file);
    } else {
      alert('Por favor, selecciona un archivo de imagen válido');
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <section className="upload-section">
      <div
        className={`upload-box ${preview ? '' : 'cursor-pointer'}`}
        onClick={handleClick}
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
        {preview ? (
          <img src={preview} alt="Preview" className="preview" />
        ) : (
          <div className="upload-content">
            <svg
              className="upload-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <h3>Sube una foto de tu piel</h3>
            <p>Haz clic aquí o arrastra una imagen</p>
            <p className="hint" style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.5rem' }}>
              Formatos: JPG, PNG (máx. 10MB)
            </p>
          </div>
        )}
      </div>
    </section>
  );
}






