/**
 * Componente Dropzone
 * ===================
 * 
 * Componente para drag and drop de archivos
 */

'use client'

import React, { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, File, X } from 'lucide-react'

export interface DropzoneProps {
  onDrop: (files: File[]) => void
  accept?: string[]
  maxSize?: number
  multiple?: boolean
  disabled?: boolean
  className?: string
  children?: React.ReactNode
}

export default function Dropzone({
  onDrop,
  accept,
  maxSize,
  multiple = false,
  disabled = false,
  className = ''
}: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const validateFile = (file: File): boolean => {
    if (accept && accept.length > 0) {
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()
      const acceptedExtensions = accept.map(ext => ext.startsWith('.') ? ext : `.${ext}`)
      
      if (!acceptedExtensions.some(ext => fileExtension === ext)) {
        return false
      }
    }

    if (maxSize && file.size > maxSize) {
      return false
    }

    return true
  }

  const handleFiles = useCallback((fileList: FileList | null) => {
    if (!fileList) return

    const validFiles = Array.from(fileList).filter(validateFile)
    
    if (validFiles.length > 0) {
      const newFiles = multiple ? [...files, ...validFiles] : validFiles
      setFiles(newFiles)
      onDrop(newFiles)
    }
  }, [files, multiple, accept, maxSize, onDrop])

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!disabled) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    if (disabled) return

    const droppedFiles = e.dataTransfer.files
    handleFiles(droppedFiles)
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files)
  }

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click()
    }
  }

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index)
    setFiles(newFiles)
  }

  return (
    <div className={className}>
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`
          relative border-2 border-dashed rounded-lg p-6
          transition-colors cursor-pointer
          ${isDragging
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
            : 'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={accept?.join(',')}
          multiple={multiple}
          onChange={handleFileInput}
          className="hidden"
          disabled={disabled}
        />

        <div className="flex flex-col items-center justify-center text-center">
          <Upload className={`w-12 h-12 mb-4 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            {isDragging ? 'Suelta los archivos aquí' : 'Arrastra archivos aquí o haz clic para seleccionar'}
          </p>
          {accept && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Formatos: {accept.join(', ')}
            </p>
          )}
          {maxSize && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Tamaño máximo: {(maxSize / 1024 / 1024).toFixed(2)} MB
            </p>
          )}
        </div>
      </div>

      {files.length > 0 && (
        <div className="mt-4 space-y-2">
          {files.map((file, index) => (
            <motion.div
              key={`${file.name}-${index}`}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex items-center gap-2 p-2 bg-gray-100 dark:bg-gray-800 rounded"
            >
              <File className="w-4 h-4 text-gray-500" />
              <span className="flex-1 text-sm text-gray-700 dark:text-gray-300 truncate">
                {file.name}
              </span>
              <span className="text-xs text-gray-500">
                {(file.size / 1024).toFixed(2)} KB
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  removeFile(index)
                }}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  )
}






