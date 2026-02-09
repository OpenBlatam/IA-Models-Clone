'use client'

import { useCallback } from 'react'
import { Checkbox } from '@/components/ui'

interface OptionsFieldsProps {
  generateTests: boolean
  includeDocker: boolean
  includeDocs: boolean
  includeCicd: boolean
  createGithubRepo: boolean
  githubPrivate: boolean
  onChange: (field: string, value: boolean) => void
}

const OptionsFields = ({
  generateTests,
  includeDocker,
  includeDocs,
  includeCicd,
  createGithubRepo,
  githubPrivate,
  onChange,
}: OptionsFieldsProps) => {
  const handleChange = useCallback(
    (field: string) => (checked: boolean) => {
      onChange(field, checked)
    },
    [onChange]
  )

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Checkbox
        label="Generate Tests"
        checked={generateTests}
        onChange={handleChange('generate_tests')}
      />
      <Checkbox
        label="Include Docker"
        checked={includeDocker}
        onChange={handleChange('include_docker')}
      />
      <Checkbox
        label="Include Documentation"
        checked={includeDocs}
        onChange={handleChange('include_docs')}
      />
      <Checkbox
        label="Include CI/CD"
        checked={includeCicd}
        onChange={handleChange('include_cicd')}
      />
      <Checkbox
        label="Create GitHub Repository"
        checked={createGithubRepo}
        onChange={handleChange('create_github_repo')}
      />
      {createGithubRepo && (
        <Checkbox
          label="Private Repository"
          checked={githubPrivate}
          onChange={handleChange('github_private')}
        />
      )}
    </div>
  )
}

export default OptionsFields

