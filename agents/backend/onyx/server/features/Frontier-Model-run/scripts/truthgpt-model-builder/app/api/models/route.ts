import { NextRequest, NextResponse } from 'next/server'
import { ModelManager } from '@/lib/modules/management'
import { logger } from '@/lib/logger'

/**
 * GET /api/models - List all models
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const limit = searchParams.get('limit') ? parseInt(searchParams.get('limit')!) : undefined
    const offset = searchParams.get('offset') ? parseInt(searchParams.get('offset')!) : undefined
    const status = searchParams.get('status')

    let models
    if (status) {
      models = ModelManager.getModelsByStatus(status as any)
    } else {
      models = await ModelManager.listModels(limit, offset)
    }

    return NextResponse.json({
      models,
      count: models.length,
      statistics: ModelManager.getStatistics(),
    })
  } catch (error) {
    logger.error('Error listing models', error instanceof Error ? error : new Error(String(error)))
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}


