import { NextResponse } from 'next/server'
import { logger } from '@/lib/logger'
import { performanceMonitor } from '@/lib/performance'
import * as fs from 'fs/promises'
import * as path from 'path'

export async function GET() {
  try {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      services: {
        logger: checkLogger(),
        performance: checkPerformance(),
        filesystem: await checkFilesystem(),
        memory: checkMemory(),
      },
    }

    // Check if any service is unhealthy
    const unhealthyServices = Object.entries(health.services).filter(
      ([, status]) => status !== 'healthy'
    )

    if (unhealthyServices.length > 0) {
      health.status = 'degraded'
      logger.warn('Health check found unhealthy services', {
        services: unhealthyServices.map(([name]) => name),
      })
    }

    const statusCode = health.status === 'healthy' ? 200 : 503

    return NextResponse.json(health, { status: statusCode })
  } catch (error) {
    logger.error('Health check failed', error instanceof Error ? error : new Error(String(error)))
    return NextResponse.json(
      {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 503 }
    )
  }
}

function checkLogger(): string {
  try {
    const logs = logger.getLogs(undefined, 1)
    return 'healthy'
  } catch {
    return 'unhealthy'
  }
}

function checkPerformance(): string {
  try {
    const metrics = performanceMonitor.export()
    return 'healthy'
  } catch {
    return 'unhealthy'
  }
}

async function checkFilesystem(): Promise<string> {
  try {
    const testPath = path.join(process.cwd(), 'package.json')
    await fs.access(testPath)
    return 'healthy'
  } catch {
    return 'unhealthy'
  }
}

function checkMemory(): string {
  const usage = process.memoryUsage()
  const maxHeapSize = 512 * 1024 * 1024 // 512 MB
  const heapUsed = usage.heapUsed

  if (heapUsed > maxHeapSize * 0.9) {
    return 'warning' // 90% of max heap
  }

  return 'healthy'
}


