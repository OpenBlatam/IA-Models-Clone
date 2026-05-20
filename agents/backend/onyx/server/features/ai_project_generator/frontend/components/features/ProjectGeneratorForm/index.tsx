'use client'

import { useState, useCallback } from 'react'
import type { ProjectRequest } from '@/types'
import { Sparkles } from 'lucide-react'
import { Button, ErrorMessage, Card } from '@/components/ui'
import { getErrorMessage } from '@/lib/utils'
import { useProjectForm } from '@/hooks/forms/useProjectForm'
import {
  DescriptionField,
  BasicInfoFields,
  PriorityField,
  OptionsFields,
  TagsField,
} from './formFields'

interface ProjectGeneratorFormProps {
  onSubmit: (request: ProjectRequest) => Promise<void>
  loading?: boolean
}

const ProjectGeneratorForm = ({ onSubmit, loading: externalLoading = false }: ProjectGeneratorFormProps) => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { formData, validationErrors, updateField, validateForm, resetForm } = useProjectForm()

  const isLoading = loading || externalLoading

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
        resetForm()
      } catch (err) {
        const errorMessage = getErrorMessage(err)
        setError(errorMessage || 'Failed to generate project')
      } finally {
        setLoading(false)
      }
    },
    [formData, onSubmit, validateForm, resetForm]
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
        <DescriptionField
          value={formData.description}
          error={validationErrors.description}
          onChange={(value) => updateField('description', value)}
        />

        <BasicInfoFields
          projectName={formData.project_name}
          author={formData.author}
          version={formData.version}
          backendFramework={formData.backend_framework}
          frontendFramework={formData.frontend_framework}
          projectNameError={validationErrors.project_name}
          onChange={updateField}
        />

        <PriorityField
          value={formData.priority}
          error={validationErrors.priority}
          onChange={(value) => updateField('priority', value)}
        />

        <OptionsFields
          generateTests={formData.generate_tests}
          includeDocker={formData.include_docker}
          includeDocs={formData.include_docs}
          includeCicd={formData.include_cicd}
          createGithubRepo={formData.create_github_repo}
          githubPrivate={formData.github_private}
          onChange={updateField}
        />

        <TagsField tags={formData.tags || []} onChange={(tags) => updateField('tags', tags)} />

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

