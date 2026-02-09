import { NextRequest, NextResponse } from 'next/server'
import { ModelManager } from '@/lib/modules/management'
import { logger } from '@/lib/logger'

/**
 * GET /api/models/[modelId] - Get model by ID
 */
export async function GET(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  try {
    const { modelId } = params

    if (!modelId) {
      return NextResponse.json(
        { error: 'Model ID is required' },
        { status: 400 }
      )
    }

    const model = await ModelManager.getModel(modelId)

    if (!model) {
      return NextResponse.json(
        { error: 'Model not found' },
        { status: 404 }
      )
    }

    return NextResponse.json({ model })
  } catch (error) {
    logger.error('Error getting model', error instanceof Error ? error : new Error(String(error)))
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

/**
 * DELETE /api/models/[modelId] - Delete model
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  try {
    const { modelId } = params

    if (!modelId) {
      return NextResponse.json(
        { error: 'Model ID is required' },
        { status: 400 }
      )
    }

    const deleted = ModelManager.deleteModel(modelId)

    if (!deleted) {
      return NextResponse.json(
        { error: 'Model not found' },
        { status: 404 }
      )
    }

    return NextResponse.json({ success: true, message: 'Model deleted' })
  } catch (error) {
    logger.error('Error deleting model', error instanceof Error ? error : new Error(String(error)))
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}


