import { NextRequest, NextResponse } from 'next/server'
import { webhookManager } from '@/lib/webhooks'
import { logger } from '@/lib/logger'

/**
 * DELETE /api/webhooks/[webhookId] - Unregister a webhook
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: { webhookId: string } }
) {
  try {
    const { webhookId } = params

    if (!webhookId) {
      return NextResponse.json(
        { error: 'Webhook ID is required' },
        { status: 400 }
      )
    }

    const deleted = webhookManager.unregister(webhookId)

    if (!deleted) {
      return NextResponse.json(
        { error: 'Webhook not found' },
        { status: 404 }
      )
    }

    return NextResponse.json({ success: true, message: 'Webhook unregistered' })
  } catch (error) {
    logger.error('Error unregistering webhook', error instanceof Error ? error : new Error(String(error)))
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}


