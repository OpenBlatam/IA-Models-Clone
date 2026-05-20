/**
 * Report Generator
 * Genera reportes en diferentes formatos (JSON, CSV, PDF-ready)
 */

import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'
import { QueueItem } from '../components/ProactiveModelBuilder'

export interface ReportData {
  title: string
  generatedAt: string
  summary: {
    totalModels: number
    successful: number
    failed: number
    successRate: number
    avgDuration: number
    totalDuration: number
  }
  models: ProactiveBuildResult[]
  queue?: QueueItem[]
  statistics?: {
    byCategory?: Record<string, number>
    byStatus?: Record<string, number>
    durationDistribution?: {
      min: number
      max: number
      p25: number
      p50: number
      p75: number
    }
  }
}

export class ReportGenerator {
  /**
   * Generar reporte completo
   */
  generateReport(
    models: ProactiveBuildResult[],
    queue?: QueueItem[],
    title: string = 'Reporte de Modelos'
  ): ReportData {
    const successful = models.filter(m => m.status === 'completed').length
    const failed = models.filter(m => m.status === 'failed').length
    const durations = models
      .filter(m => m.duration)
      .map(m => m.duration!)
    
    const avgDuration = durations.length > 0
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : 0
    
    const totalDuration = durations.reduce((a, b) => a + b, 0)

    // Distribución de duración
    const sortedDurations = [...durations].sort((a, b) => a - b)
    const p25 = this.percentile(sortedDurations, 25)
    const p50 = this.percentile(sortedDurations, 50)
    const p75 = this.percentile(sortedDurations, 75)

    // Estadísticas por categoría (si disponible)
    const byCategory: Record<string, number> = {}
    queue?.forEach(item => {
      if (item.category) {
        byCategory[item.category] = (byCategory[item.category] || 0) + 1
      }
    })

    // Estadísticas por estado
    const byStatus: Record<string, number> = {}
    models.forEach(model => {
      byStatus[model.status] = (byStatus[model.status] || 0) + 1
    })

    return {
      title,
      generatedAt: new Date().toISOString(),
      summary: {
        totalModels: models.length,
        successful,
        failed,
        successRate: models.length > 0 ? (successful / models.length) * 100 : 0,
        avgDuration: Math.round(avgDuration / 1000), // en segundos
        totalDuration: Math.round(totalDuration / 1000), // en segundos
      },
      models: [...models],
      queue: queue ? [...queue] : undefined,
      statistics: {
        byCategory: Object.keys(byCategory).length > 0 ? byCategory : undefined,
        byStatus,
        durationDistribution: sortedDurations.length > 0 ? {
          min: Math.round(Math.min(...sortedDurations) / 1000),
          max: Math.round(Math.max(...sortedDurations) / 1000),
          p25: Math.round(p25 / 1000),
          p50: Math.round(p50 / 1000),
          p75: Math.round(p75 / 1000),
        } : undefined,
      },
    }
  }

  /**
   * Exportar como JSON
   */
  exportJSON(report: ReportData, filename?: string): void {
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename || `report-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  /**
   * Exportar como CSV
   */
  exportCSV(models: ProactiveBuildResult[], filename?: string): void {
    const headers = ['Modelo', 'Estado', 'Duración (s)', 'Descripción', 'Error']
    const rows = models.map(model => [
      model.modelName,
      model.status,
      model.duration ? Math.round(model.duration / 1000).toString() : '',
      model.description,
      model.error || '',
    ])

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename || `models-${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  /**
   * Generar HTML para PDF
   */
  generateHTMLReport(report: ReportData): string {
    const stats = report.statistics
    const durationDist = stats?.durationDistribution

    return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${report.title}</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 40px;
      color: #333;
    }
    h1 { color: #8b5cf6; }
    h2 { color: #6366f1; margin-top: 30px; }
    .summary {
      background: #f3f4f6;
      padding: 20px;
      border-radius: 8px;
      margin: 20px 0;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
      margin-top: 15px;
    }
    .stat-card {
      background: white;
      padding: 15px;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stat-value {
      font-size: 24px;
      font-weight: bold;
      color: #8b5cf6;
    }
    .stat-label {
      color: #666;
      font-size: 14px;
      margin-top: 5px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 20px;
    }
    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #ddd;
    }
    th {
      background: #8b5cf6;
      color: white;
    }
    .success { color: #10b981; }
    .failed { color: #ef4444; }
  </style>
</head>
<body>
  <h1>${report.title}</h1>
  <p>Generado: ${new Date(report.generatedAt).toLocaleString()}</p>
  
  <div class="summary">
    <h2>Resumen</h2>
    <div class="summary-grid">
      <div class="stat-card">
        <div class="stat-value">${report.summary.totalModels}</div>
        <div class="stat-label">Total de Modelos</div>
      </div>
      <div class="stat-card">
        <div class="stat-value success">${report.summary.successful}</div>
        <div class="stat-label">Exitosos</div>
      </div>
      <div class="stat-card">
        <div class="stat-value failed">${report.summary.failed}</div>
        <div class="stat-label">Fallidos</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${report.summary.successRate.toFixed(1)}%</div>
        <div class="stat-label">Tasa de Éxito</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${report.summary.avgDuration}s</div>
        <div class="stat-label">Duración Promedio</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${report.summary.totalDuration}s</div>
        <div class="stat-label">Duración Total</div>
      </div>
    </div>
  </div>

  ${durationDist ? `
  <h2>Distribución de Duración</h2>
  <div class="summary-grid">
    <div class="stat-card">
      <div class="stat-value">${durationDist.min}s</div>
      <div class="stat-label">Mínimo</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${durationDist.p25}s</div>
      <div class="stat-label">Percentil 25</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${durationDist.p50}s</div>
      <div class="stat-label">Mediana</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${durationDist.p75}s</div>
      <div class="stat-label">Percentil 75</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${durationDist.max}s</div>
      <div class="stat-label">Máximo</div>
    </div>
  </div>
  ` : ''}

  <h2>Modelos Detallados</h2>
  <table>
    <thead>
      <tr>
        <th>Modelo</th>
        <th>Estado</th>
        <th>Duración</th>
        <th>Descripción</th>
        ${report.models.some(m => m.error) ? '<th>Error</th>' : ''}
      </tr>
    </thead>
    <tbody>
      ${report.models.map(model => `
        <tr>
          <td>${model.modelName}</td>
          <td class="${model.status === 'completed' ? 'success' : 'failed'}">
            ${model.status === 'completed' ? '✓ Completado' : '✗ Fallido'}
          </td>
          <td>${model.duration ? Math.round(model.duration / 1000) + 's' : '-'}</td>
          <td>${model.description}</td>
          ${report.models.some(m => m.error) ? `<td>${model.error || ''}</td>` : ''}
        </tr>
      `).join('')}
    </tbody>
  </table>
</body>
</html>
    `.trim()
  }

  /**
   * Exportar HTML para impresión/PDF
   */
  exportHTML(report: ReportData, filename?: string): void {
    const html = this.generateHTMLReport(report)
    const blob = new Blob([html], { type: 'text/html' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename || `report-${Date.now()}.html`
    a.click()
    URL.revokeObjectURL(url)
  }

  /**
   * Calcular percentil
   */
  private percentile(sortedArray: number[], percentile: number): number {
    const index = Math.ceil((percentile / 100) * sortedArray.length) - 1
    return sortedArray[Math.max(0, Math.min(index, sortedArray.length - 1))]
  }
}

// Singleton instance
let reportGeneratorInstance: ReportGenerator | null = null

export function getReportGenerator(): ReportGenerator {
  if (!reportGeneratorInstance) {
    reportGeneratorInstance = new ReportGenerator()
  }
  return reportGeneratorInstance
}

export default ReportGenerator










