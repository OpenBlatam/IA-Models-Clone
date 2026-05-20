import { NextRequest, NextResponse } from 'next/server'
import { getProactiveModelBuilder } from '@/lib/proactive-model-builder'
import { adaptiveAnalyze } from '@/lib/modules/analysis'
import { sanitizeModelName, generateModelId } from '@/lib/utils'
import { createGitHubRepository } from '@/lib/modules/github'

export async function POST(request: NextRequest) {
  try {
    const { description, optimizationCorePath } = await request.json()

    if (!description || typeof description !== 'string') {
      return NextResponse.json(
        { error: 'Se requiere una descripción del modelo' },
        { status: 400 }
      )
    }

    // Obtener path de optimization_core desde variables de entorno o parámetro
    const corePath = optimizationCorePath || 
      process.env.TRUTHGPT_OPTIMIZATION_CORE_PATH ||
      '../TruthGPT-main/optimization_core'

    // Generar ID y nombre del modelo
    const modelId = generateModelId()
    const modelName = `truthgpt-${sanitizeModelName(description)}`

    // Analizar descripción para obtener spec
    const spec = adaptiveAnalyze(description)

    // Obtener constructor proactivo
    const builder = getProactiveModelBuilder(corePath)

    // Construir modelo proactivamente (en background)
    builder.buildModelProactively({
      modelName,
      description,
      spec,
      optimizationCorePath: corePath,
      createGitHubRepo: true,
      continuous: true,
    }).then((result) => {
      console.log(`Modelo ${result.modelName} construido exitosamente`)
      console.log(`Config: ${result.configPath}`)
      console.log(`Script: ${result.scriptPath}`)
      if (result.githubUrl) {
        console.log(`GitHub: ${result.githubUrl}`)
      }
    }).catch((error) => {
      console.error(`Error construyendo modelo ${modelName}:`, error)
    })

    // Retornar respuesta inmediata
    return NextResponse.json({
      modelId,
      modelName,
      description,
      status: 'creating',
    })
  } catch (error) {
    console.error('Error in create-model-proactive API:', error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Error al crear el modelo',
      },
      { status: 500 }
    )
  }
}










