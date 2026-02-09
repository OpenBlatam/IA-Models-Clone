import { NextRequest, NextResponse } from 'next/server'
import { getModelStatus } from '@/lib/truthgpt-service'

export async function GET(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  try {
    const { modelId } = params

    if (!modelId) {
      return NextResponse.json(
        { error: 'Model ID es requerido' },
        { status: 400 }
      )
    }

    const status = await getModelStatus(modelId)

    return NextResponse.json(status)
  } catch (error) {
    console.error('Error in model-status API:', error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Error al obtener el estado',
        status: 'failed',
      },
      { status: 500 }
    )
  }
}


