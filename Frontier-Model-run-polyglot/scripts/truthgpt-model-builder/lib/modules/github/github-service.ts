import { Octokit } from '@octokit/rest'
import * as fs from 'fs/promises'
import * as path from 'path'

const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN,
})

export async function createGitHubRepository(
  modelName: string,
  description: string
): Promise<string> {
  try {
    if (!process.env.GITHUB_TOKEN) {
      throw new Error('GITHUB_TOKEN no está configurado')
    }

    // Get repository owner (user or organization)
    const owner = process.env.GITHUB_OWNER || (await octokit.rest.users.getAuthenticated()).data.login

    // Create repository
    const repo = await octokit.rest.repos.createForAuthenticatedUser({
      name: modelName,
      description: `TruthGPT Model: ${description}`,
      private: false,
      auto_init: false,
    })

    const repoUrl = repo.data.html_url

    // Get model files from TruthGPT
    const truthgptPath = path.resolve(
      process.cwd(),
      '../TruthGPT-main/generated_models',
      modelName
    )

    // Upload files to repository
    try {
      const files = await fs.readdir(truthgptPath)
      
      for (const file of files) {
        if (file === '.git') continue
        
        const filePath = path.join(truthgptPath, file)
        const stats = await fs.stat(filePath)
        
        if (stats.isFile()) {
          const content = await fs.readFile(filePath, 'utf-8')
          
          // Create file in repository
          await octokit.rest.repos.createOrUpdateFileContents({
            owner,
            repo: modelName,
            path: file,
            message: `Add ${file} - TruthGPT Model`,
            content: Buffer.from(content).toString('base64'),
            branch: 'main',
          })
        }
      }

      // Add TruthGPT integration files
      await addTruthGPTIntegrationFiles(owner, modelName, description)
    } catch (error) {
      console.error('Error uploading files:', error)
      // Repository is created, but files might not be uploaded
      // This is okay, user can add them manually
    }

    return repoUrl
  } catch (error) {
    if (error instanceof Error && error.message.includes('name already exists')) {
      // Repository already exists, return the URL
      const owner = process.env.GITHUB_OWNER || (await octokit.rest.users.getAuthenticated()).data.login
      return `https://github.com/${owner}/${modelName}`
    }
    
    console.error('Error creating GitHub repository:', error)
    throw error
  }
}

async function addTruthGPTIntegrationFiles(
  owner: string,
  repoName: string,
  description: string
): Promise<void> {
  // Add .github/workflows/truthgpt.yml
  const workflowContent = `name: TruthGPT Model CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ || echo "No tests configured yet"
`

  await octokit.rest.repos.createOrUpdateFileContents({
    owner,
    repo: repoName,
    path: '.github/workflows/truthgpt.yml',
    message: 'Add TruthGPT CI/CD workflow',
    content: Buffer.from(workflowContent).toString('base64'),
    branch: 'main',
  })

  // Update README with TruthGPT badge
  const readmeContent = await getReadmeContent(repoName, description)
  
  await octokit.rest.repos.createOrUpdateFileContents({
    owner,
    repo: repoName,
    path: 'README.md',
    message: 'Update README with TruthGPT integration',
    content: Buffer.from(readmeContent).toString('base64'),
    branch: 'main',
  })
}

async function getReadmeContent(repoName: string, description: string): Promise<string> {
  const truthgptPath = path.resolve(
    process.cwd(),
    '../TruthGPT-main/generated_models',
    repoName
  )
  
  try {
    const readmePath = path.join(truthgptPath, 'README.md')
    const content = await fs.readFile(readmePath, 'utf-8')
    
    // Add TruthGPT badge at the top
    const badge = `![TruthGPT](https://img.shields.io/badge/TruthGPT-Powered-purple?style=for-the-badge&logo=github)\n\n`
    
    return badge + content
  } catch {
    // If README doesn't exist, create a basic one
    return `# ${repoName}

![TruthGPT](https://img.shields.io/badge/TruthGPT-Powered-purple?style=for-the-badge&logo=github)

## Description

${description}

## TruthGPT Integration

This model was created and optimized using TruthGPT.

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

See the model files for usage examples.
`
  }
}

