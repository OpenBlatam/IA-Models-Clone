'use client'

import React, { useState, useEffect } from 'react'
import { Code, Github, FileCode, Sparkles, Copy, Download, Languages, Brain } from 'lucide-react'
import { Card, Button, Input, Textarea, Badge, Select } from '../ui'
import { useImproveCode, useImproveCodeText } from '@/hooks'
import { useLocalStorage } from '@/hooks'
import ExportButton from './ExportButton'
import CodeDiff from './CodeDiff'
import CodeComparison from './CodeComparison'
import ComparisonView from './ComparisonView'
import CodeMetrics from './CodeMetrics'
import FeedbackForm from './FeedbackForm'
import LoadingOverlay from './LoadingOverlay'
import ModelSelector from './ModelSelector'
import { storage, storageKeys } from '@/lib/utils'
import toast from 'react-hot-toast'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface CodeImproverProps {
  onImprovementComplete?: () => void
}

const LANGUAGES = [
  { value: 'python', label: 'Python' },
  { value: 'javascript', label: 'JavaScript' },
  { value: 'typescript', label: 'TypeScript' },
  { value: 'java', label: 'Java' },
  { value: 'cpp', label: 'C++' },
  { value: 'c', label: 'C' },
  { value: 'go', label: 'Go' },
  { value: 'rust', label: 'Rust' },
  { value: 'php', label: 'PHP' },
  { value: 'ruby', label: 'Ruby' },
]

const CodeImprover: React.FC<CodeImproverProps> = ({
  onImprovementComplete,
}) => {
  const [method, setMethod] = useState<'github' | 'text'>('github')
  const [result, setResult] = useState<{
    original: string
    improved: string
    suggestions: Array<{
      type: string
      description: string
      line?: number
      severity?: string
    }>
  } | null>(null)
  const [language, setLanguage] = useState('python')
  const [showDiff, setShowDiff] = useState(false)
  const [showComparison, setShowComparison] = useState(false)
  const [showComparisonView, setShowComparisonView] = useState(false)
  const [showFeedback, setShowFeedback] = useState(false)
  const [showModelSelector, setShowModelSelector] = useState(false)
  const [settings] = useLocalStorage('settings', { defaultModel: '' })

  // GitHub method
  const [githubRepo, setGithubRepo] = useState('')
  const [filePath, setFilePath] = useState('')
  const [branch, setBranch] = useState('')
  const [modelId, setModelId] = useState(settings?.defaultModel || '')

  // Text method
  const [codeText, setCodeText] = useState('')
  const [context, setContext] = useState('')

  const improveCodeMutation = useImproveCode()
  const improveCodeTextMutation = useImproveCodeText()

  // Save to history when improvement completes
  useEffect(() => {
    if (result) {
      const history = storage.get<Array<any>>(storageKeys.CODE_HISTORY, [])
      const newItem = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        repo: githubRepo || undefined,
        filePath: filePath || undefined,
        originalCode: result.original,
        improvedCode: result.improved,
        improvementsCount: result.suggestions.length,
        language,
      }
      const updatedHistory = [newItem, ...(history || []).slice(0, 49)] // Keep last 50
      storage.set(storageKeys.CODE_HISTORY, updatedHistory)
      setShowFeedback(true)
    }
  }, [result, githubRepo, filePath, language])

  const handleGitHubImprove = async () => {
    if (!githubRepo || !filePath) {
      toast.error('Please provide repository and file path')
      return
    }

    try {
      const response = await improveCodeMutation.mutateAsync({
        github_repo: githubRepo,
        file_path: filePath,
        branch: branch || undefined,
        model_id: modelId || settings?.defaultModel || undefined,
      })

      setResult({
        original: response.original_code,
        improved: response.improved_code,
        suggestions: response.suggestions,
      })
      onImprovementComplete?.()
    } catch (error) {
      // Error handled by mutation
    }
  }

  const handleTextImprove = async () => {
    if (!codeText.trim()) {
      toast.error('Please provide code to improve')
      return
    }

    try {
      const response = await improveCodeTextMutation.mutateAsync({
        code: codeText,
        context: context || undefined,
        modelId: modelId || settings?.defaultModel || undefined,
      })

      setResult({
        original: response.original_code,
        improved: response.improved_code,
        suggestions: response.suggestions,
      })
      onImprovementComplete?.()
    } catch (error) {
      // Error handled by mutation
    }
  }

  const handleCopy = (text: string, label: string) => {
    navigator.clipboard.writeText(text)
    toast.success(`${label} copied to clipboard!`)
  }

  const handleDownload = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    toast.success('File downloaded!')
  }

  const detectLanguage = (code: string): string => {
    if (code.includes('def ') || code.includes('import ') || code.includes('print(')) {
      return 'python'
    }
    if (code.includes('function ') || code.includes('const ') || code.includes('let ')) {
      if (code.includes(': ') && code.includes('interface ')) {
        return 'typescript'
      }
      return 'javascript'
    }
    if (code.includes('public class') || code.includes('public static')) {
      return 'java'
    }
    if (code.includes('#include') || code.includes('std::')) {
      return 'cpp'
    }
    return 'python' // default
  }

  const handleCodeChange = (newCode: string) => {
    setCodeText(newCode)
    if (newCode.trim()) {
      const detected = detectLanguage(newCode)
      setLanguage(detected)
    }
  }

  const handleFeedbackSubmit = (feedback: { rating: number; comment: string }) => {
    // Save feedback to localStorage or send to API
    const feedbacks = storage.get<Array<any>>('feedback', [])
    feedbacks.push({
      ...feedback,
      timestamp: new Date().toISOString(),
      resultId: result ? 'current' : null,
    })
    storage.set('feedback', feedbacks)
    setShowFeedback(false)
  }

  const isImproving = improveCodeMutation.isPending || improveCodeTextMutation.isPending

  return (
    <div className="space-y-6 relative">
      <LoadingOverlay isLoading={isImproving} message="Improving your code..." />

      <Card>
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Code className="w-6 h-6 text-primary-600" />
            <h2 className="text-xl font-semibold">Improve Code</h2>
          </div>

          <div className="flex gap-4 border-b border-gray-200 pb-4">
            <button
              onClick={() => setMethod('github')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                method === 'github'
                  ? 'bg-primary-100 text-primary-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
              disabled={isImproving}
            >
              <Github className="w-4 h-4 inline mr-2" />
              From GitHub
            </button>
            <button
              onClick={() => setMethod('text')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                method === 'text'
                  ? 'bg-primary-100 text-primary-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
              disabled={isImproving}
            >
              <FileCode className="w-4 h-4 inline mr-2" />
              From Text
            </button>
          </div>

          {method === 'github' ? (
            <div className="space-y-4">
              <Input
                label="GitHub Repository"
                placeholder="owner/repo"
                value={githubRepo}
                onChange={(e) => setGithubRepo(e.target.value)}
                disabled={isImproving}
                helperText="Format: username/repository"
              />
              <Input
                label="File Path"
                placeholder="src/main.py"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
                disabled={isImproving}
              />
              <Input
                label="Branch (optional)"
                placeholder="main"
                value={branch}
                onChange={(e) => setBranch(e.target.value)}
                disabled={isImproving}
              />
              <div className="flex gap-2">
                <Input
                  label="Model ID (optional)"
                  placeholder={settings?.defaultModel || "Leave empty for default"}
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  disabled={isImproving}
                  className="flex-1"
                />
                <div className="pt-6">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowModelSelector(!showModelSelector)}
                    disabled={isImproving}
                    title="Select Model"
                  >
                    <Brain className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              {showModelSelector && (
                <ModelSelector
                  selectedModelId={modelId || undefined}
                  onSelect={(id) => {
                    setModelId(id || '')
                    setShowModelSelector(false)
                  }}
                />
              )}
              <Button
                onClick={handleGitHubImprove}
                isLoading={isImproving}
                className="w-full"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Improve Code
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <Select
                  label="Language"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  options={LANGUAGES}
                  disabled={isImproving}
                  className="flex-1"
                />
                <div className="pt-6">
                  <Languages className="w-5 h-5 text-gray-400" />
                </div>
              </div>
              <Textarea
                label="Code"
                placeholder="Paste your code here..."
                value={codeText}
                onChange={(e) => handleCodeChange(e.target.value)}
                disabled={isImproving}
                rows={12}
                className="font-mono text-sm"
              />
              <Textarea
                label="Context (optional)"
                placeholder="Additional context about the code..."
                value={context}
                onChange={(e) => setContext(e.target.value)}
                disabled={isImproving}
                rows={3}
              />
              <div className="flex gap-2">
                <Input
                  label="Model ID (optional)"
                  placeholder={settings?.defaultModel || "Leave empty for default"}
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  disabled={isImproving}
                  className="flex-1"
                />
                <div className="pt-6">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowModelSelector(!showModelSelector)}
                    disabled={isImproving}
                    title="Select Model"
                  >
                    <Brain className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              {showModelSelector && (
                <ModelSelector
                  selectedModelId={modelId || undefined}
                  onSelect={(id) => {
                    setModelId(id || '')
                    setShowModelSelector(false)
                  }}
                />
              )}
              <Button
                onClick={handleTextImprove}
                isLoading={isImproving}
                className="w-full"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Improve Code
              </Button>
            </div>
          )}
        </div>
      </Card>

      {result && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Results</h3>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowComparisonView(!showComparisonView)}
              >
                {showComparisonView ? 'Hide' : 'Show'} Comparison View
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowDiff(!showDiff)}
              >
                {showDiff ? 'Hide' : 'Show'} Diff View
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowComparison(!showComparison)}
              >
                {showComparison ? 'Hide' : 'Show'} Comparison
              </Button>
            </div>
          </div>

          {showComparisonView && (
            <ComparisonView
              original={result.original}
              improved={result.improved}
              suggestions={result.suggestions}
              language={language}
            />
          )}

          {showComparison && !showComparisonView && (
            <CodeComparison
              original={result.original}
              improved={result.improved}
              language={language}
              suggestions={result.suggestions}
            />
          )}

          {showDiff && !showComparisonView && (
            <CodeDiff
              original={result.original}
              improved={result.improved}
              language={language}
            />
          )}

          <CodeMetrics
            original={result.original}
            improved={result.improved}
            suggestions={result.suggestions}
          />

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Original Code</h3>
                <div className="flex gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCopy(result.original, 'Original code')}
                  >
                    <Copy className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDownload(result.original, 'original.txt')}
                  >
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <SyntaxHighlighter
                  language={language}
                  style={vscDarkPlus}
                  customStyle={{ borderRadius: '8px', fontSize: '14px' }}
                >
                  {result.original}
                </SyntaxHighlighter>
              </div>
            </Card>

            <Card>
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Improved Code</h3>
                <div className="flex gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCopy(result.improved, 'Improved code')}
                  >
                    <Copy className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDownload(result.improved, 'improved.txt')}
                  >
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <SyntaxHighlighter
                  language={language}
                  style={vscDarkPlus}
                  customStyle={{ borderRadius: '8px', fontSize: '14px' }}
                >
                  {result.improved}
                </SyntaxHighlighter>
              </div>
            </Card>
          </div>

          {result.suggestions.length > 0 && (
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">
                  Suggestions ({result.suggestions.length})
                </h3>
                <ExportButton
                  data={{
                    original: result.original,
                    improved: result.improved,
                    suggestions: result.suggestions,
                    metadata: {
                      timestamp: new Date().toISOString(),
                      language,
                    },
                  }}
                  filename="code-improvement"
                />
              </div>
              <div className="space-y-3">
                {result.suggestions.map((suggestion, index) => (
                  <div
                    key={index}
                    className="p-4 bg-gray-50 rounded-lg border border-gray-200 hover:border-primary-300 transition-colors"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Badge
                        variant={
                          suggestion.severity === 'high'
                            ? 'error'
                            : suggestion.severity === 'medium'
                            ? 'warning'
                            : 'info'
                        }
                        size="sm"
                      >
                        {suggestion.type}
                      </Badge>
                      {suggestion.line && (
                        <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded">
                          Line {suggestion.line}
                        </span>
                      )}
                      {suggestion.severity && (
                        <span className="text-xs text-gray-500">
                          {suggestion.severity} priority
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-700 leading-relaxed">
                      {suggestion.description}
                    </p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {showFeedback && (
            <FeedbackForm
              onSubmit={handleFeedbackSubmit}
              onCancel={() => setShowFeedback(false)}
            />
          )}
        </div>
      )}
    </div>
  )
}

export default CodeImprover
