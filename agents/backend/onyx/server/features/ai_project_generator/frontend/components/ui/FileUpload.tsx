'use client'

import { useState, useCallback, useRef, ReactNode } from 'react'
import { Upload, X, File as FileIcon, CheckCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button, Progress } from '@/components/ui'
import { fileUtils } from '@/lib/utils'

interface FileUploadProps {
  onUpload?: (files: File[]) => void
  accept?: string
  multiple?: boolean
  maxSize?: number
  maxFiles?: number
  className?: string
  children?: ReactNode
  showPreview?: boolean
}

const FileUpload = ({
  onUpload,
  accept,
  multiple = false,
  maxSize,
  maxFiles,
  className,
  children,
  showPreview = true,
}: FileUploadProps) => {
  const [files, setFiles] = useState<File[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(
    (newFiles: FileList | null) => {
      if (!newFiles) {
        return
      }

      const fileArray = Array.from(newFiles)
      let validFiles: File[] = []

      fileArray.forEach((file) => {
        if (maxSize && file.size > maxSize) {
          alert(`File ${file.name} exceeds maximum size of ${fileUtils.getFileSize(maxSize)}`)
          return
        }

        if (accept && !fileUtils.validateFileType(file, accept.split(','))) {
          alert(`File ${file.name} is not an accepted file type`)
          return
        }

        validFiles.push(file)
      })

      if (maxFiles && files.length + validFiles.length > maxFiles) {
        alert(`Maximum ${maxFiles} files allowed`)
        return
      }

      const updatedFiles = multiple ? [...files, ...validFiles] : validFiles
      setFiles(updatedFiles)
      onUpload?.(updatedFiles)

      validFiles.forEach((file) => {
        setUploadProgress((prev) => ({ ...prev, [file.name]: 100 }))
      })
    },
    [files, multiple, maxSize, accept, maxFiles, onUpload]
  )

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setDragActive(false)
      handleFiles(e.dataTransfer.files)
    },
    [handleFiles]
  )

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFiles(e.target.files)
    },
    [handleFiles]
  )

  const removeFile = useCallback(
    (fileName: string) => {
      const updatedFiles = files.filter((f) => f.name !== fileName)
      setFiles(updatedFiles)
      setUploadProgress((prev) => {
        const newProgress = { ...prev }
        delete newProgress[fileName]
        return newProgress
      })
      onUpload?.(updatedFiles)
    },
    [files, onUpload]
  )

  const openFileDialog = useCallback(() => {
    inputRef.current?.click()
  }, [])

  return (
    <div className={cn('space-y-4', className)}>
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={cn(
          'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
          dragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-gray-400'
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleChange}
          className="hidden"
        />
        {children || (
          <>
            <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
            <p className="text-gray-600 mb-2">
              Drag and drop files here, or{' '}
              <button
                onClick={openFileDialog}
                className="text-primary-600 hover:text-primary-700 underline"
              >
                browse
              </button>
            </p>
            {accept && <p className="text-sm text-gray-500">Accepted: {accept}</p>}
            {maxSize && (
              <p className="text-sm text-gray-500">Max size: {fileUtils.getFileSize(maxSize)}</p>
            )}
          </>
        )}
      </div>

      {showPreview && files.length > 0 && (
        <div className="space-y-2">
          {files.map((file) => (
            <div
              key={file.name}
              className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg"
            >
              <FileIcon className="w-5 h-5 text-gray-400 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">{file.name}</p>
                <p className="text-xs text-gray-500">{fileUtils.getFileSize(file.size)}</p>
                {uploadProgress[file.name] !== undefined && (
                  <Progress
                    value={uploadProgress[file.name]}
                    size="sm"
                    className="mt-1"
                    showLabel={false}
                  />
                )}
              </div>
              {uploadProgress[file.name] === 100 && (
                <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
              )}
              <button
                onClick={() => removeFile(file.name)}
                className="p-1 hover:bg-gray-200 rounded flex-shrink-0"
                aria-label={`Remove ${file.name}`}
              >
                <X className="w-4 h-4 text-gray-400" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default FileUpload

