'use client'

import React, { useState } from 'react'
import { Github, Search, FileCode, AlertCircle, CheckCircle } from 'lucide-react'
import { Card, Button, Input, Badge, LoadingSpinner } from '../ui'
import { apiClient } from '@/lib/api'
import toast from 'react-hot-toast'
import type { RepositoryAnalyzeResponse } from '@/lib/api/types'

interface RepositoryAnalyzerProps {
  onAnalysisComplete?: (results: RepositoryAnalyzeResponse) => void
}

const RepositoryAnalyzer: React.FC<RepositoryAnalyzerProps> = ({
  onAnalysisComplete,
}) => {
  const [githubRepo, setGithubRepo] = useState('')
  const [branch, setBranch] = useState('')
  const [modelId, setModelId] = useState('')
  const [maxFiles, setMaxFiles] = useState(10)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<RepositoryAnalyzeResponse | null>(null)

  const handleAnalyze = async () => {
    if (!githubRepo) {
      toast.error('Please provide a GitHub repository')
      return
    }

    setIsAnalyzing(true)
    try {
      const response = await apiClient.analyzeRepository({
        github_repo: githubRepo,
        branch: branch || undefined,
        model_id: modelId || undefined,
        max_files: maxFiles,
      })

      setResults(response)
      toast.success(
        `Analysis complete! ${response.files_analyzed} files analyzed with ${response.total_improvements} improvements.`
      )
      onAnalysisComplete?.(response)
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : 'Failed to analyze repository'
      )
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Github className="w-6 h-6 text-primary-600" />
            <h2 className="text-xl font-semibold">Analyze Repository</h2>
          </div>

          <div className="space-y-4">
            <Input
              label="GitHub Repository"
              placeholder="owner/repo"
              value={githubRepo}
              onChange={(e) => setGithubRepo(e.target.value)}
              disabled={isAnalyzing}
              helperText="Format: username/repository"
            />
            <Input
              label="Branch (optional)"
              placeholder="main"
              value={branch}
              onChange={(e) => setBranch(e.target.value)}
              disabled={isAnalyzing}
            />
            <Input
              label="Model ID (optional)"
              placeholder="Leave empty for default"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={isAnalyzing}
            />
            <Input
              label="Max Files"
              type="number"
              min="1"
              max="50"
              value={maxFiles}
              onChange={(e) => setMaxFiles(parseInt(e.target.value) || 10)}
              disabled={isAnalyzing}
              helperText="Maximum number of files to analyze (1-50)"
            />
            <Button
              onClick={handleAnalyze}
              isLoading={isAnalyzing}
              className="w-full"
            >
              <Search className="w-4 h-4 mr-2" />
              Analyze Repository
            </Button>
          </div>
        </div>
      </Card>

      {results && (
        <Card>
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">
                Analysis Results
              </h3>
              <Badge variant="success" size="lg">
                {results.total_improvements} improvements
              </Badge>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
                  <FileCode className="w-4 h-4" />
                  <span>Files Analyzed</span>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {results.files_analyzed}
                </div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-2 text-sm text-gray-600 mb-1">
                  <CheckCircle className="w-4 h-4" />
                  <span>Total Improvements</span>
                </div>
                <div className="text-2xl font-bold text-green-600">
                  {results.total_improvements}
                </div>
              </div>
            </div>

            {results.improvements.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">
                  File Improvements
                </h4>
                <div className="space-y-3">
                  {results.improvements.map((improvement, index) => (
                    <div
                      key={index}
                      className="p-4 border border-gray-200 rounded-lg hover:border-primary-300 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-mono text-sm font-medium text-gray-900">
                          {improvement.file_path}
                        </span>
                        <Badge variant="info" size="sm">
                          {improvement.suggestions?.length || 0} suggestions
                        </Badge>
                      </div>
                      {improvement.suggestions && improvement.suggestions.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {improvement.suggestions.slice(0, 3).map((suggestion, idx) => (
                            <div
                              key={idx}
                              className="text-xs text-gray-600 flex items-center gap-2"
                            >
                              <AlertCircle className="w-3 h-3" />
                              <span>{suggestion.description}</span>
                            </div>
                          ))}
                          {improvement.suggestions.length > 3 && (
                            <p className="text-xs text-gray-500 mt-1">
                              +{improvement.suggestions.length - 3} more suggestions
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  )
}

export default RepositoryAnalyzer



