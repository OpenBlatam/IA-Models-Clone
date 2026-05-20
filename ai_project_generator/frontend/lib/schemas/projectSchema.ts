import { z } from 'zod'

export const projectRequestSchema = z.object({
  description: z.string().min(10, 'Description must be at least 10 characters').max(2000),
  project_name: z.string().min(3, 'Project name must be at least 3 characters').max(50).optional(),
  author: z.string().optional(),
  version: z.string().regex(/^\d+\.\d+\.\d+$/, 'Version must be in format X.Y.Z').optional(),
  priority: z.number().min(-10).max(10).default(0),
  backend_framework: z.enum(['fastapi', 'flask', 'django']).default('fastapi'),
  frontend_framework: z.enum(['react', 'vue', 'nextjs']).default('react'),
  generate_tests: z.boolean().default(true),
  include_docker: z.boolean().default(true),
  include_docs: z.boolean().default(true),
  include_cicd: z.boolean().default(true),
  create_github_repo: z.boolean().default(false),
  github_private: z.boolean().default(false),
  tags: z.array(z.string()).default([]),
})

export type ProjectRequestSchema = z.infer<typeof projectRequestSchema>

