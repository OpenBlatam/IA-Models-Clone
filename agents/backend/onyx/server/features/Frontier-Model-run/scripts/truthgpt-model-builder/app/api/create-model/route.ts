import { NextRequest, NextResponse } from 'next/server'
import { createTruthGPTModel } from '@/lib/truthgpt-service'
import { createGitHubRepository } from '@/lib/modules/github'
import { sanitizeModelName, generateModelId } from '@/lib/utils'

export async function POST(request: NextRequest) {
  try {
    const { description } = await request.json()

    if (!description || typeof description !== 'string') {
      return NextResponse.json(
        { error: 'Se requiere una descripción del modelo' },
        { status: 400 }
      )
    }

    // Generate model ID and name
    const modelId = generateModelId()
    const modelName = `truthgpt-${sanitizeModelName(description)}`

    // Create TruthGPT model (async - will be processed in background)
    createTruthGPTModel(modelId, modelName, description).catch((error) => {
      console.error('Error creating TruthGPT model:', error)
    })

    // Create GitHub repository immediately
    const githubUrl = await createGitHubRepository(modelName, description)

    return NextResponse.json({
      modelId,
      modelName,
      description,
      githubUrl,
      status: 'creating',
    })
  } catch (error) {
    console.error('Error in create-model API:', error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Error al crear el modelo',
      },
      { status: 500 }
    )
  }
}

