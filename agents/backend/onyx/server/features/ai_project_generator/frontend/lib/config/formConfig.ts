import type { ProjectRequest } from '@/types'

export const INITIAL_FORM_DATA: ProjectRequest = {
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

export const BACKEND_FRAMEWORKS = [
  { value: 'fastapi', label: 'FastAPI' },
  { value: 'flask', label: 'Flask' },
  { value: 'django', label: 'Django' },
] as const

export const FRONTEND_FRAMEWORKS = [
  { value: 'react', label: 'React' },
  { value: 'vue', label: 'Vue' },
  { value: 'nextjs', label: 'Next.js' },
] as const

export const FORM_VALIDATION = {
  description: {
    minLength: 10,
    maxLength: 2000,
  },
  projectName: {
    minLength: 3,
    maxLength: 50,
  },
  priority: {
    min: -10,
    max: 10,
  },
  version: {
    pattern: /^\d+\.\d+\.\d+$/,
  },
} as const

