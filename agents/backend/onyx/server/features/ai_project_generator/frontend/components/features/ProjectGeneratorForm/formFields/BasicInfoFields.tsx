'use client'

import { useCallback } from 'react'
import { Input, Select } from '@/components/ui'
import { BACKEND_FRAMEWORKS, FRONTEND_FRAMEWORKS } from '@/lib/config/formConfig'

interface BasicInfoFieldsProps {
  projectName: string
  author: string
  version: string
  backendFramework: string
  frontendFramework: string
  projectNameError?: string
  onChange: (field: string, value: string) => void
}

const BasicInfoFields = ({
  projectName,
  author,
  version,
  backendFramework,
  frontendFramework,
  projectNameError,
  onChange,
}: BasicInfoFieldsProps) => {
  const handleChange = useCallback(
    (field: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      onChange(field, e.target.value)
    },
    [onChange]
  )

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Input
        label="Project Name"
        value={projectName}
        onChange={handleChange('project_name')}
        placeholder="my-ai-project"
        error={projectNameError}
      />
      <Input
        label="Author"
        value={author}
        onChange={handleChange('author')}
        placeholder="Your Name"
      />
      <Input
        label="Version"
        value={version}
        onChange={handleChange('version')}
        placeholder="1.0.0"
      />
      <Select
        label="Backend Framework"
        value={backendFramework}
        onChange={handleChange('backend_framework')}
        options={BACKEND_FRAMEWORKS}
      />
      <Select
        label="Frontend Framework"
        value={frontendFramework}
        onChange={handleChange('frontend_framework')}
        options={FRONTEND_FRAMEWORKS}
      />
    </div>
  )
}

export default BasicInfoFields

