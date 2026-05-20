import { prisma } from './prisma';
import { Redis } from 'ioredis';
import { CircuitBreaker } from './circuit-breaker';

export interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  uptime: number;
  version: string;
  services: Record<string, ServiceHealth>;
}

export interface ServiceHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTime: number;
  error?: string;
  lastCheck: string;
}

export class HealthChecker {
  private redis: Redis | null = null;
  private startTime: number = Date.now();

  constructor() {
    console.log('Redis disabled for production health checks');
  }

  async checkHealth(): Promise<HealthCheckResult> {
    const timestamp = new Date().toISOString();
    const uptime = Date.now() - this.startTime;
    const version = process.env.npm_package_version || '1.0.0';

    const services: Record<string, ServiceHealth> = {};

    const checks = [
      { name: 'database', check: () => this.checkDatabase() },
      { name: 'redis', check: () => this.checkRedis() },
      { name: 'openai', check: () => this.checkOpenAI() },
      { name: 'stripe', check: () => this.checkStripe() },
      { name: 'memory', check: () => this.checkMemory() },
      { name: 'disk', check: () => this.checkDisk() }
    ];

    await Promise.allSettled(
      checks.map(async ({ name, check }) => {
        try {
          services[name] = await check();
        } catch (error) {
          services[name] = {
            status: 'unhealthy',
            responseTime: 0,
            error: error instanceof Error ? error.message : 'Unknown error',
            lastCheck: timestamp
          };
        }
      })
    );

    const overallStatus = this.determineOverallStatus(services);

    return {
      status: overallStatus,
      timestamp,
      uptime,
      version,
      services
    };
  }

  private async checkDatabase(): Promise<ServiceHealth> {
    const start = Date.now();
    
    try {
      await prisma.$queryRaw`SELECT 1`;
      const responseTime = Date.now() - start;
      
      return {
        status: responseTime < 1000 ? 'healthy' : 'degraded',
        responseTime,
        lastCheck: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - start,
        error: error instanceof Error ? error.message : 'Database connection failed',
        lastCheck: new Date().toISOString()
      };
    }
  }

  private async checkRedis(): Promise<ServiceHealth> {
    if (!this.redis) {
      return {
        status: 'degraded',
        responseTime: 0,
        error: 'Redis not configured',
        lastCheck: new Date().toISOString()
      };
    }

    const start = Date.now();
    
    try {
      await this.redis.ping();
      const responseTime = Date.now() - start;
      
      return {
        status: responseTime < 500 ? 'healthy' : 'degraded',
        responseTime,
        lastCheck: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - start,
        error: error instanceof Error ? error.message : 'Redis connection failed',
        lastCheck: new Date().toISOString()
      };
    }
  }

  private async checkOpenAI(): Promise<ServiceHealth> {
    const start = Date.now();
    
    try {
      if (!process.env.OPENAI_API_KEY) {
        return {
          status: 'degraded',
          responseTime: 0,
          error: 'OpenAI API key not configured',
          lastCheck: new Date().toISOString()
        };
      }

      const response = await fetch('https://api.openai.com/v1/models', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        },
        signal: AbortSignal.timeout(5000)
      });

      const responseTime = Date.now() - start;
      
      return {
        status: response.ok && responseTime < 3000 ? 'healthy' : 'degraded',
        responseTime,
        lastCheck: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - start,
        error: error instanceof Error ? error.message : 'OpenAI API check failed',
        lastCheck: new Date().toISOString()
      };
    }
  }

  private async checkStripe(): Promise<ServiceHealth> {
    const start = Date.now();
    
    try {
      if (!process.env.STRIPE_SECRET_KEY) {
        return {
          status: 'degraded',
          responseTime: 0,
          error: 'Stripe secret key not configured',
          lastCheck: new Date().toISOString()
        };
      }

      const response = await fetch('https://api.stripe.com/v1/account', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${process.env.STRIPE_SECRET_KEY}`,
        },
        signal: AbortSignal.timeout(5000)
      });

      const responseTime = Date.now() - start;
      
      return {
        status: response.ok && responseTime < 2000 ? 'healthy' : 'degraded',
        responseTime,
        lastCheck: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: Date.now() - start,
        error: error instanceof Error ? error.message : 'Stripe API check failed',
        lastCheck: new Date().toISOString()
      };
    }
  }

  private checkMemory(): ServiceHealth {
    const memUsage = process.memoryUsage();
    const totalMem = memUsage.heapTotal;
    const usedMem = memUsage.heapUsed;
    const memoryUsagePercent = (usedMem / totalMem) * 100;

    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (memoryUsagePercent > 90) status = 'unhealthy';
    else if (memoryUsagePercent > 75) status = 'degraded';

    return {
      status,
      responseTime: 0,
      lastCheck: new Date().toISOString(),
      ...(status !== 'healthy' && { 
        error: `Memory usage at ${memoryUsagePercent.toFixed(1)}%` 
      })
    };
  }

  private checkDisk(): ServiceHealth {
    try {
      const fs = require('fs');
      const stats = fs.statSync('.');
      
      return {
        status: 'healthy',
        responseTime: 0,
        lastCheck: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        responseTime: 0,
        error: 'Disk check failed',
        lastCheck: new Date().toISOString()
      };
    }
  }

  private determineOverallStatus(services: Record<string, ServiceHealth>): 'healthy' | 'degraded' | 'unhealthy' {
    const statuses = Object.values(services).map(s => s.status);
    
    if (statuses.includes('unhealthy')) {
      const unhealthyCount = statuses.filter(s => s === 'unhealthy').length;
      const criticalServices = ['database'];
      const hasCriticalFailure = Object.entries(services).some(
        ([name, service]) => criticalServices.includes(name) && service.status === 'unhealthy'
      );
      
      if (hasCriticalFailure || unhealthyCount > 2) {
        return 'unhealthy';
      }
      return 'degraded';
    }
    
    if (statuses.includes('degraded')) {
      return 'degraded';
    }
    
    return 'healthy';
  }
}

export const healthChecker = new HealthChecker();
