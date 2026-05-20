/**
 * Model Exporter
 * Sistema mejorado de exportación de modelos
 */

import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'

export type ExportFormat = 'json' | 'csv' | 'yaml' | 'markdown' | 'html' | 'pdf'

export interface ExportOptions {
  format: ExportFormat
  includeMetadata?: boolean
  includeConfig?: boolean
  includeMetrics?: boolean
  includeHistory?: boolean
  filter?: (model: ProactiveBuildResult) => boolean
}

export class ModelExporter {
  /**
   * Exportar modelos
   */
  async exportModels(
    models: ProactiveBuildResult[],
    options: ExportOptions
  ): Promise<Blob> {
    const filtered = options.filter
      ? models.filter(options.filter)
      : models

    switch (options.format) {
      case 'json':
        return this.exportJSON(filtered, options)
      case 'csv':
        return this.exportCSV(filtered, options)
      case 'yaml':
        return this.exportYAML(filtered, options)
      case 'markdown':
        return this.exportMarkdown(filtered, options)
      case 'html':
        return this.exportHTML(filtered, options)
      case 'pdf':
        return this.exportPDF(filtered, options)
      default:
        throw new Error(`Formato no soportado: ${options.format}`)
    }
  }

  /**
   * Exportar JSON
   */
  private exportJSON(models: ProactiveBuildResult[], options: ExportOptions): Blob {
    const data = models.map(model => ({
      modelId: model.modelId,
      modelName: model.modelName,
      description: model.description,
      status: model.status,
      duration: model.duration,
      error: model.error,
      startTime: model.startTime,
      endTime: model.endTime,
      ...(options.includeMetadata && {
        createdAt: model.startTime,
            updatedAt: model.endTime,
          }),
    }))

    return new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    })
  }

  /**
   * Exportar CSV
   */
  private exportCSV(models: ProactiveBuildResult[], options: ExportOptions): Blob {
    const headers = [
      'Model ID',
      'Model Name',
      'Description',
      'Status',
      'Duration (ms)',
      'Duration (s)',
      'Error',
      'Start Time',
      'End Time',
    ]

    const rows = models.map(model => [
      model.modelId,
      model.modelName,
      `"${model.description.replace(/"/g, '""')}"`,
      model.status,
      model.duration?.toString() || '',
      model.duration ? (model.duration / 1000).toFixed(2) : '',
      model.error ? `"${model.error.replace(/"/g, '""')}"` : '',
      model.startTime ? new Date(model.startTime).toISOString() : '',
      model.endTime ? new Date(model.endTime).toISOString() : '',
    ])

    const csv = [
      headers.join(','),
      ...rows.map(row => row.join(',')),
    ].join('\n')

    return new Blob([csv], { type: 'text/csv' })
  }

  /**
   * Exportar YAML
   */
  private exportYAML(models: ProactiveBuildResult[], options: ExportOptions): Blob {
    const yaml = models.map(model => {
      const lines = [
        `- modelId: ${model.modelId}`,
        `  modelName: ${model.modelName}`,
        `  description: "${model.description.replace(/"/g, '\\"')}"`,
        `  status: ${model.status}`,
      ]

      if (model.duration) {
        lines.push(`  duration: ${model.duration}`)
        lines.push(`  durationSeconds: ${(model.duration / 1000).toFixed(2)}`)
      }

      if (model.error) {
        lines.push(`  error: "${model.error.replace(/"/g, '\\"')}"`)
      }

      if (model.startTime) {
        lines.push(`  startTime: ${model.startTime}`)
      }

      if (model.endTime) {
        lines.push(`  endTime: ${model.endTime}`)
      }

      return lines.join('\n')
    }).join('\n\n')

    return new Blob([yaml], { type: 'text/yaml' })
  }

  /**
   * Exportar Markdown
   */
  private exportMarkdown(models: ProactiveBuildResult[], options: ExportOptions): Blob {
    const md = [
      '# Modelos Exportados',
      `\nFecha de exportación: ${new Date().toISOString()}`,
      `\nTotal de modelos: ${models.length}`,
      '\n## Lista de Modelos\n',
      '| Model ID | Name | Status | Duration | Error |',
      '|----------|------|--------|----------|-------|',
      ...models.map(model => {
        const duration = model.duration
          ? `${(model.duration / 1000).toFixed(2)}s`
          : 'N/A'
        const error = model.error ? `\`${model.error.substring(0, 50)}\`` : '-'
        return `| ${model.modelId} | ${model.modelName} | ${model.status} | ${duration} | ${error} |`
      }),
    ].join('\n')

    return new Blob([md], { type: 'text/markdown' })
  }

  /**
   * Exportar HTML
   */
  private exportHTML(models: ProactiveBuildResult[], options: ExportOptions): Blob {
    const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Modelos Exportados</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #333; }
    th { background: #9333EA; color: white; }
    tr:hover { background: #2a2a2a; }
    .success { color: #10b981; }
    .failed { color: #ef4444; }
    .stats { background: #2a2a2a; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
  </style>
</head>
<body>
  <h1>Modelos Exportados</h1>
  <div class="stats">
    <p><strong>Fecha:</strong> ${new Date().toLocaleString()}</p>
    <p><strong>Total:</strong> ${models.length}</p>
    <p><strong>Exitosos:</strong> ${models.filter(m => m.status === 'completed').length}</p>
    <p><strong>Fallidos:</strong> ${models.filter(m => m.status === 'failed').length}</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>Model ID</th>
        <th>Name</th>
        <th>Status</th>
        <th>Duration</th>
        <th>Error</th>
      </tr>
    </thead>
    <tbody>
      ${models.map(model => {
        const duration = model.duration
          ? `${(model.duration / 1000).toFixed(2)}s`
          : 'N/A'
        const statusClass = model.status === 'completed' ? 'success' : 'failed'
        return `
        <tr>
          <td>${model.modelId}</td>
          <td>${model.modelName}</td>
          <td class="${statusClass}">${model.status}</td>
          <td>${duration}</td>
          <td>${model.error || '-'}</td>
        </tr>
        `
      }).join('')}
    </tbody>
  </table>
</body>
</html>
    `.trim()

    return new Blob([html], { type: 'text/html' })
  }

  /**
   * Exportar PDF (usando HTML como base)
   */
  private async exportPDF(models: ProactiveBuildResult[], options: ExportOptions): Promise<Blob> {
    // Por ahora, exportamos HTML que puede ser convertido a PDF
    // En producción, usarías una librería como jsPDF o html2pdf
    const html = await this.exportHTML(models, options)
    return html
  }

  /**
   * Descargar archivo
   */
  downloadFile(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }
}

// Singleton instance
let modelExporterInstance: ModelExporter | null = null

export function getModelExporter(): ModelExporter {
  if (!modelExporterInstance) {
    modelExporterInstance = new ModelExporter()
  }
  return modelExporterInstance
}

export default ModelExporter










