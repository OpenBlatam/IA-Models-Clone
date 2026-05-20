/**
 * Sistema de Reporting para Tests E2E
 * 
 * Genera reportes en diferentes formatos (texto, HTML, JSON)
 */
import { TestMetrics } from './metrics';
import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

export interface TestReport {
  testName: string;
  status: 'PASSED' | 'FAILED';
  duration: number;
  durationFormatted: string;
  steps: Array<{
    name: string;
    duration: number;
    durationFormatted: string;
    status: 'PASSED' | 'FAILED';
  }>;
  performance?: {
    pageLoadTime: number;
    totalRequests: number;
    failedRequests: number;
    averageResponseTime: number;
  };
  screenshots: string[];
  errors: string[];
  timestamp: string;
}

// ============================================================================
// Report Generation
// ============================================================================

/**
 * Genera un reporte estructurado desde métricas
 */
export function generateReport(metrics: TestMetrics): TestReport {
  const report: TestReport = {
    testName: metrics.testName,
    status: metrics.passed ? 'PASSED' : 'FAILED',
    duration: metrics.totalDuration,
    durationFormatted: formatDuration(metrics.totalDuration),
    steps: metrics.steps.map((step) => ({
      name: step.name,
      duration: step.duration,
      durationFormatted: formatDuration(step.duration),
      status: step.passed ? 'PASSED' : 'FAILED',
    })),
    screenshots: metrics.screenshots,
    errors: metrics.errors,
    timestamp: new Date().toISOString(),
  };

  if (metrics.performance) {
    report.performance = {
      pageLoadTime: metrics.performance.pageLoadTime,
      totalRequests: metrics.performance.totalRequests,
      failedRequests: metrics.performance.failedRequests,
      averageResponseTime: metrics.performance.averageResponseTime,
    };
  }

  return report;
}

/**
 * Genera reporte en formato texto
 */
export function generateTextReport(report: TestReport): string {
  const lines: string[] = [];

  lines.push('='.repeat(60));
  lines.push(`Test: ${report.testName}`);
  lines.push('='.repeat(60));
  lines.push(`Status: ${report.status === 'PASSED' ? '✅ PASSED' : '❌ FAILED'}`);
  lines.push(`Duration: ${report.durationFormatted}`);
  lines.push(`Timestamp: ${report.timestamp}`);
  lines.push('');

  if (report.steps.length > 0) {
    lines.push('Steps:');
    report.steps.forEach((step) => {
      const icon = step.status === 'PASSED' ? '✅' : '❌';
      lines.push(`  ${icon} ${step.name} (${step.durationFormatted})`);
    });
    lines.push('');
  }

  if (report.performance) {
    lines.push('Performance Metrics:');
    lines.push(`  Page Load Time: ${formatDuration(report.performance.pageLoadTime)}`);
    lines.push(`  Total Requests: ${report.performance.totalRequests}`);
    lines.push(`  Failed Requests: ${report.performance.failedRequests}`);
    lines.push(`  Avg Response Time: ${formatDuration(report.performance.averageResponseTime)}`);
    lines.push('');
  }

  if (report.screenshots.length > 0) {
    lines.push('Screenshots:');
    report.screenshots.forEach((screenshot) => {
      lines.push(`  - ${screenshot}`);
    });
    lines.push('');
  }

  if (report.errors.length > 0) {
    lines.push('Errors:');
    report.errors.forEach((error) => {
      lines.push(`  ❌ ${error}`);
    });
    lines.push('');
  }

  lines.push('='.repeat(60));

  return lines.join('\n');
}

/**
 * Genera reporte en formato HTML
 */
export function generateHtmlReport(report: TestReport): string {
  const statusColor = report.status === 'PASSED' ? '#28a745' : '#dc3545';
  const statusIcon = report.status === 'PASSED' ? '✅' : '❌';

  return `
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Test Report: ${report.testName}</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      background: #f5f5f5;
    }
    .container {
      background: white;
      border-radius: 8px;
      padding: 30px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
      color: #333;
      border-bottom: 3px solid ${statusColor};
      padding-bottom: 10px;
    }
    .status {
      display: inline-block;
      padding: 8px 16px;
      border-radius: 4px;
      color: white;
      background: ${statusColor};
      font-weight: bold;
      margin: 10px 0;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 15px;
      margin: 20px 0;
    }
    .metric {
      background: #f8f9fa;
      padding: 15px;
      border-radius: 4px;
      border-left: 4px solid ${statusColor};
    }
    .metric-label {
      font-size: 12px;
      color: #666;
      text-transform: uppercase;
    }
    .metric-value {
      font-size: 24px;
      font-weight: bold;
      color: #333;
    }
    .steps {
      margin: 20px 0;
    }
    .step {
      padding: 10px;
      margin: 5px 0;
      border-radius: 4px;
      border-left: 4px solid #ddd;
    }
    .step.passed {
      background: #d4edda;
      border-color: #28a745;
    }
    .step.failed {
      background: #f8d7da;
      border-color: #dc3545;
    }
    .error {
      background: #f8d7da;
      color: #721c24;
      padding: 10px;
      border-radius: 4px;
      margin: 5px 0;
    }
    .screenshot {
      margin: 10px 0;
    }
    .screenshot img {
      max-width: 100%;
      border: 1px solid #ddd;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>${statusIcon} ${report.testName}</h1>
    <div class="status">${report.status}</div>
    
    <div class="metrics">
      <div class="metric">
        <div class="metric-label">Duration</div>
        <div class="metric-value">${report.durationFormatted}</div>
      </div>
      ${report.performance ? `
      <div class="metric">
        <div class="metric-label">Page Load</div>
        <div class="metric-value">${formatDuration(report.performance.pageLoadTime)}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Requests</div>
        <div class="metric-value">${report.performance.totalRequests}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Failed</div>
        <div class="metric-value">${report.performance.failedRequests}</div>
      </div>
      ` : ''}
    </div>

    ${report.steps.length > 0 ? `
    <div class="steps">
      <h2>Steps</h2>
      ${report.steps.map(step => `
        <div class="step ${step.status.toLowerCase()}">
          <strong>${step.status === 'PASSED' ? '✅' : '❌'}</strong> ${step.name}
          <span style="float: right; color: #666;">${step.durationFormatted}</span>
        </div>
      `).join('')}
    </div>
    ` : ''}

    ${report.errors.length > 0 ? `
    <div>
      <h2>Errors</h2>
      ${report.errors.map(error => `
        <div class="error">❌ ${error}</div>
      `).join('')}
    </div>
    ` : ''}

    ${report.screenshots.length > 0 ? `
    <div>
      <h2>Screenshots</h2>
      ${report.screenshots.map(screenshot => `
        <div class="screenshot">
          <img src="${screenshot}" alt="Screenshot" />
          <div>${screenshot}</div>
        </div>
      `).join('')}
    </div>
    ` : ''}

    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
      Generated at: ${report.timestamp}
    </div>
  </div>
</body>
</html>
  `.trim();
}

/**
 * Genera reporte en formato JSON
 */
export function generateJsonReport(report: TestReport): string {
  return JSON.stringify(report, null, 2);
}

// ============================================================================
// File Operations
// ============================================================================

/**
 * Guarda un reporte en un archivo
 */
export async function saveReport(
  report: TestReport,
  format: 'html' | 'json' | 'txt' = 'html',
  outputDir: string = 'test-results'
): Promise<string> {
  // Crear directorio si no existe
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Generar contenido según formato
  let content: string;
  let extension: string;

  switch (format) {
    case 'html':
      content = generateHtmlReport(report);
      extension = 'html';
      break;
    case 'json':
      content = generateJsonReport(report);
      extension = 'json';
      break;
    case 'txt':
      content = generateTextReport(report);
      extension = 'txt';
      break;
  }

  // Generar nombre de archivo
  const timestamp = Date.now();
  const safeTestName = report.testName.replace(/[^a-z0-9]/gi, '-').toLowerCase();
  const filename = `${safeTestName}-${timestamp}.${extension}`;
  const filepath = path.join(outputDir, filename);

  // Escribir archivo
  fs.writeFileSync(filepath, content, 'utf-8');

  return filepath;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Formatea duración en milisegundos a string legible
 */
function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(2)}s`;
  } else {
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(2);
    return `${minutes}m ${seconds}s`;
  }
}
