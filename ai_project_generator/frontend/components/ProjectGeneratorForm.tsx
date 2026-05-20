'use client'

import { useState, useCallback } from 'react'
import clsx from 'clsx'
import type { ProjectRequest } from '@/types'
import { Sparkles, HelpCircle } from 'lucide-react'
import ErrorMessage from './ErrorMessage'
import { getErrorMessage } from '@/lib/utils'
import Tooltip from './Tooltip'

interface ProjectGeneratorFormProps {
  onSubmit: (request: ProjectRequest) => Promise<void>
}

const ProjectGeneratorForm = ({ onSubmit }: ProjectGeneratorFormProps) => {
  const [formData, setFormData] = useState<ProjectRequest>({
    description: '',
    project_name: '',
    author: 'Blatam Academy',
    version: '1.0.0',
    priority: 0,
    backend_framework: 'fastapi',
    frontend_framework: 'react',
    generate_tests: true,
    include_docker: true,
    include_docs: true,
    include_cicd: true,
    create_github_repo: false,
    github_private: false,
    tags: [],
  })

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tagInput, setTagInput] = useState('')

  const handleChange = useCallback((field: keyof ProjectRequest, value: unknown) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    setError(null)
  }, [])

  const handleAddTag = useCallback(() => {
    if (tagInput.trim() && !formData.tags?.includes(tagInput.trim())) {
      handleChange('tags', [...(formData.tags || []), tagInput.trim()])
      setTagInput('')
    }
  }, [tagInput, formData.tags, handleChange])

  const handleRemoveTag = useCallback((tag: string) => {
    handleChange(
      'tags',
      formData.tags?.filter((t) => t !== tag) || []
    )
  }, [formData.tags, handleChange])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      await onSubmit(formData)
      setFormData({
        description: '',
        project_name: '',
        author: 'Blatam Academy',
        version: '1.0.0',
        priority: 0,
        backend_framework: 'fastapi',
        frontend_framework: 'react',
        generate_tests: true,
        include_docker: true,
        include_docs: true,
        include_cicd: true,
        create_github_repo: false,
        github_private: false,
        tags: [],
      })
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      setError(errorMessage || 'Failed to generate project')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.target instanceof HTMLInputElement) {
      e.preventDefault()
      handleAddTag()
    }
  }, [handleAddTag])

  return (
    <div className="card max-w-4xl mx-auto">
      <div className="flex items-center gap-2 mb-6">
        <Sparkles className="w-6 h-6 text-primary-600" />
        <h2 className="text-2xl font-bold text-gray-900">Generate New Project</h2>
      </div>

      {error && (
        <div className="mb-4">
          <ErrorMessage message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <label htmlFor="description" className="block text-sm font-medium text-gray-700">
              Project Description *
            </label>
            <Tooltip content="Provide a detailed description of your AI project. Include the type of AI (chat, vision, audio, etc.) and key features you need.">
              <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" aria-hidden="true" />
            </Tooltip>
          </div>
          <textarea
            id="description"
            value={formData.description}
            onChange={(e) => handleChange('description', e.target.value)}
            required
            minLength={10}
            maxLength={2000}
            rows={5}
            className="input"
            placeholder="Describe your AI project in detail..."
            aria-label="Project Description"
          />
          <p className="mt-1 text-sm text-gray-500">
            {formData.description.length}/2000 characters
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label htmlFor="project_name" className="block text-sm font-medium text-gray-700 mb-2">
              Project Name
            </label>
            <input
              id="project_name"
              type="text"
              value={formData.project_name}
              onChange={(e) => handleChange('project_name', e.target.value)}
              minLength={3}
              maxLength={50}
              className="input"
              placeholder="my_ai_project"
              aria-label="Project Name"
            />
          </div>

          <div>
            <label htmlFor="author" className="block text-sm font-medium text-gray-700 mb-2">
              Author
            </label>
            <input
              id="author"
              type="text"
              value={formData.author}
              onChange={(e) => handleChange('author', e.target.value)}
              className="input"
              placeholder="Your Name"
              aria-label="Author"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label htmlFor="version" className="block text-sm font-medium text-gray-700 mb-2">
              Version
            </label>
            <input
              id="version"
              type="text"
              value={formData.version}
              onChange={(e) => handleChange('version', e.target.value)}
              className="input"
              placeholder="1.0.0"
              aria-label="Version"
            />
          </div>

          <div>
            <div className="flex items-center gap-2 mb-2">
              <label htmlFor="priority" className="block text-sm font-medium text-gray-700">
                Priority (-10 to 10)
              </label>
              <Tooltip content="Higher priority projects are processed first. Use positive numbers for urgent projects.">
                <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" aria-hidden="true" />
              </Tooltip>
            </div>
            <input
              id="priority"
              type="number"
              value={formData.priority}
              onChange={(e) => handleChange('priority', parseInt(e.target.value, 10) || 0)}
              min={-10}
              max={10}
              className="input"
              aria-label="Priority"
            />
          </div>

          <div>
            <label htmlFor="backend_framework" className="block text-sm font-medium text-gray-700 mb-2">
              Backend Framework
            </label>
            <select
              id="backend_framework"
              value={formData.backend_framework}
              onChange={(e) => handleChange('backend_framework', e.target.value)}
              className="input"
              aria-label="Backend Framework"
            >
              <option value="fastapi">FastAPI</option>
              <option value="flask">Flask</option>
              <option value="django">Django</option>
            </select>
          </div>
        </div>

        <div>
          <label htmlFor="frontend_framework" className="block text-sm font-medium text-gray-700 mb-2">
            Frontend Framework
          </label>
          <select
            id="frontend_framework"
            value={formData.frontend_framework}
            onChange={(e) => handleChange('frontend_framework', e.target.value)}
            className="input"
            aria-label="Frontend Framework"
          >
            <option value="react">React</option>
            <option value="vue">Vue</option>
            <option value="nextjs">Next.js</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Options
          </label>
          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData.generate_tests}
                onChange={(e) => handleChange('generate_tests', e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                aria-label="Generate Tests"
              />
              <span className="ml-2 text-sm text-gray-700">Generate Tests</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData.include_docker}
                onChange={(e) => handleChange('include_docker', e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                aria-label="Include Docker"
              />
              <span className="ml-2 text-sm text-gray-700">Include Docker</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData.include_docs}
                onChange={(e) => handleChange('include_docs', e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                aria-label="Include Documentation"
              />
              <span className="ml-2 text-sm text-gray-700">Include Documentation</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData.include_cicd}
                onChange={(e) => handleChange('include_cicd', e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                aria-label="Include CI/CD"
              />
              <span className="ml-2 text-sm text-gray-700">Include CI/CD</span>
            </label>
          </div>
        </div>

        <div>
          <label htmlFor="tags" className="block text-sm font-medium text-gray-700 mb-2">
            Tags
          </label>
          <div className="flex gap-2 mb-2">
            <input
              id="tags"
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="input flex-1"
              placeholder="Add a tag and press Enter"
              aria-label="Add Tag"
            />
            <button
              type="button"
              onClick={handleAddTag}
              className="btn btn-secondary"
              tabIndex={0}
              aria-label="Add Tag Button"
            >
              Add
            </button>
          </div>
          {formData.tags && formData.tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {formData.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 px-3 py-1 bg-primary-100 text-primary-800 rounded-full text-sm"
                >
                  {tag}
                  <button
                    type="button"
                    onClick={() => handleRemoveTag(tag)}
                    className="text-primary-600 hover:text-primary-800"
                    tabIndex={0}
                    aria-label={`Remove tag ${tag}`}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-4">
          <button
            type="submit"
            disabled={loading || !formData.description.trim()}
            className={clsx(
              'btn btn-primary',
              (loading || !formData.description.trim()) &&
                'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            tabIndex={0}
            aria-label="Generate Project"
            aria-disabled={loading || !formData.description.trim()}
          >
            {loading ? 'Generating...' : 'Generate Project'}
          </button>
        </div>
      </form>
    </div>
  )
}

export default ProjectGeneratorForm

