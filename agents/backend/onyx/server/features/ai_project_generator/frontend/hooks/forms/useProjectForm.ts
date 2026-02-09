import { useState, useCallback } from 'react'
import type { ProjectRequest } from '@/types'
import { INITIAL_FORM_DATA } from '@/lib/config/formConfig'
import { validateDescription, validateProjectName, validatePriority } from '@/lib/utils'

export const useProjectForm = () => {
  const [formData, setFormData] = useState<ProjectRequest>(INITIAL_FORM_DATA)
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})

  const updateField = useCallback((field: keyof ProjectRequest, value: unknown) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    setValidationErrors((prev) => {
      const newErrors = { ...prev }
      delete newErrors[field]
      return newErrors
    })
  }, [])

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

  const resetForm = useCallback(() => {
    setFormData(INITIAL_FORM_DATA)
    setValidationErrors({})
  }, [])

  return {
    formData,
    validationErrors,
    updateField,
    validateForm,
    resetForm,
    setFormData,
  }
}

