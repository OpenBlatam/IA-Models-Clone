import { NextRequest, NextResponse } from 'next/server'
import { webhookManager } from '@/lib/webhooks'
import { logger } from '@/lib/logger'

/**
 * POST /api/webhooks - Register a webhook
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, url, secret, events, timeout } = body

    if (!id || !url || !events || !Array.isArray(events)) {
      return NextResponse.json(
        { error: 'Missing required fields: id, url, events' },
        { status: 400 }
      )
    }

    // Validate URL
    try {
      new URL(url)
    } catch {
      return NextResponse.json(
        { error: 'Invalid URL format' },
        { status: 400 }
      )
    }

    webhookManager.register(id, {
      url,
      secret,
      events,
      timeout,
    })

    return NextResponse.json({ success: true, message: 'Webhook registered' })
  } catch (error) {
    logger.error('Error registering webhook', error instanceof Error ? error : new Error(String(error)))
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

/**
 * GET /api/webhooks - List all webhooks
 */
export async function GET() {
  try {
    const webhooks = webhookManager.list()
    return NextResponse.json({ webhooks })
  } catch (error) {
    logger.error('Error listing webhooks', error instanceof Error ? error : new Error(String(error)))
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}


