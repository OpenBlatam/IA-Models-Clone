'use client'

import { useState, useCallback } from 'react'
import type { ProjectRequest } from '@/types'
import { Sparkles, Plus, X } from 'lucide-react'
import { Button, Input, Textarea, Select, Checkbox, ErrorMessage, Tooltip, Card, HelpTooltip } from '@/components/ui'
import { getErrorMessage, validateDescription, validateProjectName, validatePriority } from '@/lib/utils'

interface ProjectGeneratorFormProps {
  onSubmit: (request: ProjectRequest) => Promise<void>
  loading?: boolean
}

const INITIAL_FORM_DATA: ProjectRequest = {
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
}

const BACKEND_FRAMEWORKS = [
  { value: 'fastapi', label: 'FastAPI' },
  { value: 'flask', label: 'Flask' },
  { value: 'django', label: 'Django' },
]

const FRONTEND_FRAMEWORKS = [
  { value: 'react', label: 'React' },
  { value: 'vue', label: 'Vue' },
  { value: 'nextjs', label: 'Next.js' },
]

const ProjectGeneratorForm = ({ onSubmit, loading: externalLoading = false }: ProjectGeneratorFormProps) => {
  const [formData, setFormData] = useState<ProjectRequest>(INITIAL_FORM_DATA)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tagInput, setTagInput] = useState('')
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})

  const isLoading = loading || externalLoading

  const validateForm = useCallback((): boolean => {
    const errors: Record<string, string> = {}

    const descValidation = validateDescription(formData.description)
    if (!descValidation.valid) {
      errors.description = descValidation.error || 'Invalid description'
    }

    if (formData.project_name) {
      const nameValidation = validateProjectName(formData.project_name)
      if (!nameValidation.valid) {
        errors.project_name = nameValidation.error || 'Invalid project name'
      }
    }

    const priorityValidation = validatePriority(formData.priority)
    if (!priorityValidation.valid) {
      errors.priority = priorityValidation.error || 'Invalid priority'
    }

    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }, [formData])

  const handleChange = useCallback((field: keyof ProjectRequest, value: unknown) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    setError(null)
    if (validationErrors[field]) {
      setValidationErrors((prev) => {
        const newErrors = { ...prev }
        delete newErrors[field]
        return newErrors
      })
    }
  }, [validationErrors])

  const handleAddTag = useCallback(() => {
    const trimmedTag = tagInput.trim()
    if (trimmedTag && !formData.tags?.includes(trimmedTag)) {
      handleChange('tags', [...(formData.tags || []), trimmedTag])
      setTagInput('')
    }
  }, [tagInput, formData.tags, handleChange])

  const handleRemoveTag = useCallback(
    (tag: string) => {
      handleChange(
        'tags',
        formData.tags?.filter((t) => t !== tag) || []
      )
    },
    [formData.tags, handleChange]
  )

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      setError(null)

      if (!validateForm()) {
        return
      }

      setLoading(true)

      try {
        await onSubmit(formData)
        setFormData(INITIAL_FORM_DATA)
        setTagInput('')
        setValidationErrors({})
      } catch (err) {
        const errorMessage = getErrorMessage(err)
        setError(errorMessage || 'Failed to generate project')
      } finally {
        setLoading(false)
      }
    },
    [formData, onSubmit, validateForm]
  )

  const handleTagKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        handleAddTag()
      }
    },
    [handleAddTag]
  )

  return (
    <Card className="max-w-4xl mx-auto">
      <div className="flex items-center gap-2 mb-6">
        <Sparkles className="w-6 h-6 text-primary-600" aria-hidden="true" />
        <h2 className="text-2xl font-bold text-gray-900">Generate New Project</h2>
      </div>

      {error && (
        <div className="mb-4">
          <ErrorMessage message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6" noValidate>
        <div className="relative">
          <div className="flex items-center gap-2 mb-2">
            <label htmlFor="description" className="block text-sm font-medium text-gray-700">
              Project Description *
            </label>
            <HelpTooltip content="Provide a detailed description of your AI project. Include the type of AI (chat, vision, audio, etc.) and key features you need." />
          </div>
          <Textarea
            id="description"
            value={formData.description}
            onChange={(e) => handleChange('description', e.target.value)}
            required
            minLength={10}
            maxLength={2000}
            rows={5}
            placeholder="Describe your AI project in detail..."
            error={validationErrors.description}
            showCharCount
            helperText="Include the type of AI (chat, vision, audio, etc.) and key features you need."
            className="w-full"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Project Name"
            value={formData.project_name}
            onChange={(e) => handleChange('project_name', e.target.value)}
            minLength={3}
            maxLength={50}
            placeholder="my_ai_project"
            error={validationErrors.project_name}
            helperText="Optional: Auto-generated if not provided"
          />

          <Input
            label="Author"
            value={formData.author}
            onChange={(e) => handleChange('author', e.target.value)}
            placeholder="Your Name"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Input
            label="Version"
            value={formData.version}
            onChange={(e) => handleChange('version', e.target.value)}
            placeholder="1.0.0"
          />

          <div className="relative">
            <div className="flex items-center gap-2 mb-2">
              <label htmlFor="priority" className="block text-sm font-medium text-gray-700">
                Priority (-10 to 10)
              </label>
              <HelpTooltip content="Higher priority projects are processed first. Use positive numbers for urgent projects." />
            </div>
            <Input
              id="priority"
              type="number"
              value={formData.priority}
              onChange={(e) => handleChange('priority', parseInt(e.target.value, 10) || 0)}
              min={-10}
              max={10}
              error={validationErrors.priority}
              helperText="Higher priority = processed first"
              className="w-full"
            />
          </div>

          <Select
            label="Backend Framework"
            value={formData.backend_framework}
            onChange={(e) => handleChange('backend_framework', e.target.value)}
            options={BACKEND_FRAMEWORKS}
          />
        </div>

        <Select
          label="Frontend Framework"
          value={formData.frontend_framework}
          onChange={(e) => handleChange('frontend_framework', e.target.value)}
          options={FRONTEND_FRAMEWORKS}
        />

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Options</label>
          <div className="space-y-2">
            <Checkbox
              label="Generate Tests"
              checked={formData.generate_tests}
              onChange={(e) => handleChange('generate_tests', e.target.checked)}
            />
            <Checkbox
              label="Include Docker"
              checked={formData.include_docker}
              onChange={(e) => handleChange('include_docker', e.target.checked)}
            />
            <Checkbox
              label="Include Documentation"
              checked={formData.include_docs}
              onChange={(e) => handleChange('include_docs', e.target.checked)}
            />
            <Checkbox
              label="Include CI/CD"
              checked={formData.include_cicd}
              onChange={(e) => handleChange('include_cicd', e.target.checked)}
            />
          </div>
        </div>

        <div>
          <label htmlFor="tags" className="block text-sm font-medium text-gray-700 mb-2">
            Tags
          </label>
          <div className="flex gap-2 mb-2">
            <Input
              id="tags"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={handleTagKeyDown}
              placeholder="Add a tag and press Enter"
              className="flex-1"
            />
            <Button
              type="button"
              variant="secondary"
              size="md"
              leftIcon={<Plus className="w-4 h-4" />}
              onClick={handleAddTag}
              disabled={!tagInput.trim()}
            >
              Add
            </Button>
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
                    className="text-primary-600 hover:text-primary-800 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded"
                    tabIndex={0}
                    aria-label={`Remove tag ${tag}`}
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-4">
          <Button
            type="submit"
            variant="primary"
            size="lg"
            isLoading={isLoading}
            disabled={!formData.description.trim()}
          >
            Generate Project
          </Button>
        </div>
      </form>
    </Card>
  )
}

export default ProjectGeneratorForm

