'use client'

import React, { useState, useCallback } from 'react'
import { Upload, FileText, Link as LinkIcon } from 'lucide-react'
import { Button, Card, Input, Modal } from '../ui'
import FileUploader from './FileUploader'
import { apiClient } from '@/lib/api'
import toast from 'react-hot-toast'

interface PaperUploadProps {
  onUploadSuccess?: () => void
}

const PaperUpload: React.FC<PaperUploadProps> = ({ onUploadSuccess }) => {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [uploadMethod, setUploadMethod] = useState<'file' | 'link'>('file')
  const [linkUrl, setLinkUrl] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)

  const handleFileUpload = useCallback(
    async (file: File) => {
      setIsProcessing(true)
      try {
        await apiClient.uploadPaper(file)
        toast.success('Paper uploaded successfully!')
        setIsModalOpen(false)
        onUploadSuccess?.()
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : 'Failed to upload paper'
        )
        throw error
      } finally {
        setIsProcessing(false)
      }
    },
    [onUploadSuccess]
  )

  const handleLinkSubmit = async () => {
    if (!linkUrl.trim()) {
      toast.error('Please enter a valid URL')
      return
    }

    setIsProcessing(true)
    try {
      await apiClient.processLink({ url: linkUrl, download: true })
      toast.success('Paper processed successfully!')
      setLinkUrl('')
      setIsModalOpen(false)
      onUploadSuccess?.()
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : 'Failed to process link'
      )
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <>
      <Button onClick={() => setIsModalOpen(true)}>
        <Upload className="w-4 h-4 mr-2" />
        Upload Paper
      </Button>

      <Modal
        isOpen={isModalOpen}
        onClose={() => !isProcessing && setIsModalOpen(false)}
        title="Upload Research Paper"
        size="lg"
      >
        <div className="space-y-6">
          <div className="flex gap-4 border-b border-gray-200 pb-4">
            <button
              onClick={() => setUploadMethod('file')}
              disabled={isProcessing}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                uploadMethod === 'file'
                  ? 'bg-primary-100 text-primary-700'
                  : 'text-gray-600 hover:bg-gray-100'
              } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <FileText className="w-4 h-4 inline mr-2" />
              Upload PDF
            </button>
            <button
              onClick={() => setUploadMethod('link')}
              disabled={isProcessing}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                uploadMethod === 'link'
                  ? 'bg-primary-100 text-primary-700'
                  : 'text-gray-600 hover:bg-gray-100'
              } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <LinkIcon className="w-4 h-4 inline mr-2" />
              From URL
            </button>
          </div>

          {uploadMethod === 'file' ? (
            <FileUploader
              onUpload={handleFileUpload}
              accept={{ 'application/pdf': ['.pdf'] }}
              maxSize={50 * 1024 * 1024}
              maxFiles={1}
              label="Upload PDF File"
              helperText="Only PDF files are supported. Maximum size: 50MB"
            />
          ) : (
            <div className="space-y-4">
              <Input
                label="Paper URL"
                type="url"
                placeholder="https://arxiv.org/pdf/..."
                value={linkUrl}
                onChange={(e) => setLinkUrl(e.target.value)}
                disabled={isProcessing}
                helperText="Enter a URL to a research paper PDF"
              />
              <Button
                onClick={handleLinkSubmit}
                isLoading={isProcessing}
                className="w-full"
              >
                Process Paper
              </Button>
            </div>
          )}
        </div>
      </Modal>
    </>
  )
}

export default PaperUpload
