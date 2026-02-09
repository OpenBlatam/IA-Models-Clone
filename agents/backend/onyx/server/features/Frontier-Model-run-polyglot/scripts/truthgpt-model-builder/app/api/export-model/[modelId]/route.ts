import { NextRequest, NextResponse } from 'next/server'
import { getModelHistory } from '@/lib/storage'

export async function GET(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  try {
    const { modelId } = params
    const history = getModelHistory()
    const model = history.find(m => m.id === modelId)

    if (!model) {
      return NextResponse.json(
        { error: 'Model not found' },
        { status: 404 }
      )
    }

    const format = request.nextUrl.searchParams.get('format') || 'json'

    if (format === 'json') {
      return NextResponse.json(model, {
        headers: {
          'Content-Disposition': `attachment; filename="${model.name}-config.json"`,
        },
      })
    }

    // Add other formats as needed
    return NextResponse.json(model)
  } catch (error) {
    console.error('Error exporting model:', error)
    return NextResponse.json(
      { error: 'Error exporting model' },
      { status: 500 }
    )
  }
}


