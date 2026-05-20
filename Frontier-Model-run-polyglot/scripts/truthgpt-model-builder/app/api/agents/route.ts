import { NextRequest, NextResponse } from 'next/server'
import { createTruthGPTModel } from '@/lib/truthgpt-service'
import { createGitHubRepository } from '@/lib/modules/github'
import { sanitizeModelName, generateModelId } from '@/lib/utils'

export async function POST(request: NextRequest) {
  try {
    const { name, description } = await request.json()

    if (!name || typeof name !== 'string') {
      return NextResponse.json(
        { error: 'Se requiere un nombre para el agente' },
        { status: 400 }
      )
    }

    if (!description || typeof description !== 'string') {
      return NextResponse.json(
        { error: 'Se requiere una descripción del agente' },
        { status: 400 }
      )
    }

    // Generate model ID and agent name
    const modelId = generateModelId()
    const agentName = `agent-${sanitizeModelName(name)}`

    // Create TruthGPT model/agent (async - will be processed in background)
    createTruthGPTModel(modelId, agentName, description).catch((error) => {
      console.error('Error creating agent:', error)
    })

    // Create GitHub repository immediately
    let githubUrl = null
    try {
      githubUrl = await createGitHubRepository(agentName, description)
    } catch (error) {
      console.error('Error creating GitHub repository:', error)
      // Continue even if GitHub creation fails
    }

    return NextResponse.json({
      modelId,
      modelName: agentName,
      agentName: name,
      description,
      githubUrl,
      status: 'completed',
    })
  } catch (error) {
    console.error('Error in create-agent API:', error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Error al crear el agente',
      },
      { status: 500 }
    )
  }
}

/**
 * GET /api/agents - List all agents
 */
export async function GET(request: NextRequest) {
  try {
    const { ModelManager } = await import('@/lib/model-manager')
    const { logger } = await import('@/lib/logger')

    const searchParams = request.nextUrl.searchParams
    const limit = searchParams.get('limit') ? parseInt(searchParams.get('limit')!) : undefined
    const offset = searchParams.get('offset') ? parseInt(searchParams.get('offset')!) : undefined

    // Get all models that are agents (name starts with 'agent-')
    const allModels = await ModelManager.listModels(limit, offset)
    const agents = allModels.filter((model: any) => model.name?.startsWith('agent-'))

    return NextResponse.json({
      agents,
      count: agents.length,
    })
  } catch (error) {
    console.error('Error listing agents:', error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Error desconocido',
      },
      { status: 500 }
    )
  }
}


