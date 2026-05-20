/**
 * Utilidades para Manejo de Errores en Tests
 * 
 * Proporciona funciones para manejar errores de forma consistente y reportar problemas
 */
import { Page } from '@playwright/test';
import { takeNamedScreenshot } from '../test-helpers';

// ============================================================================
// Types
// ============================================================================

export interface ErrorContext {
  testName: string;
  step: string;
  error: Error;
  page?: Page;
  timestamp: number;
  screenshots?: string[];
  networkErrors?: Array<{ url: string; status: number }>;
  consoleErrors?: string[];
}

export interface ErrorReport {
  context: ErrorContext;
  stackTrace: string;
  suggestions: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
}

// ============================================================================
// Error Collection
// ============================================================================

/**
 * Recolecta información completa del contexto de error
 */
export async function collectErrorContext(
  testName: string,
  step: string,
  error: Error,
  page?: Page
): Promise<ErrorContext> {
  const context: ErrorContext = {
    testName,
    step,
    error,
    timestamp: Date.now(),
  };

  if (page) {
    context.page = page;

    // Capturar screenshot
    try {
      const screenshot = await takeNamedScreenshot(
        page,
        `error-${testName}-${step}`
      );
      context.screenshots = [screenshot];
    } catch (screenshotError) {
      console.warn('No se pudo capturar screenshot:', screenshotError);
    }

    // Recolectar errores de consola
    try {
      const consoleMessages: string[] = [];
      page.on('console', (msg) => {
        if (msg.type() === 'error') {
          consoleMessages.push(msg.text());
        }
      });
      context.consoleErrors = consoleMessages;
    } catch (consoleError) {
      console.warn('No se pudieron recolectar errores de consola:', consoleError);
    }
  }

  return context;
}

// ============================================================================
// Error Analysis
// ============================================================================

/**
 * Analiza un error y genera sugerencias
 */
export function analyzeError(context: ErrorContext): ErrorReport {
  const { error, step } = context;
  const errorMessage = error.message.toLowerCase();
  const suggestions: string[] = [];
  let severity: ErrorReport['severity'] = 'medium';

  // Analizar tipo de error
  if (errorMessage.includes('timeout') || errorMessage.includes('timed out')) {
    severity = 'high';
    suggestions.push('Verificar que el servidor está corriendo');
    suggestions.push('Aumentar timeout si es necesario');
    suggestions.push('Verificar conectividad de red');
  } else if (errorMessage.includes('not found') || errorMessage.includes('404')) {
    severity = 'high';
    suggestions.push('Verificar que la ruta existe');
    suggestions.push('Verificar configuración de rutas');
  } else if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
    severity = 'critical';
    suggestions.push('Verificar conectividad de red');
    suggestions.push('Verificar que el servidor está accesible');
    suggestions.push('Revisar logs del servidor');
  } else if (errorMessage.includes('element') || errorMessage.includes('selector')) {
    severity = 'medium';
    suggestions.push('Verificar que el selector es correcto');
    suggestions.push('Verificar que el elemento existe en el DOM');
    suggestions.push('Aumentar tiempo de espera si el elemento se carga dinámicamente');
  } else if (errorMessage.includes('assertion') || errorMessage.includes('expect')) {
    severity = 'low';
    suggestions.push('Verificar los valores esperados');
    suggestions.push('Revisar la lógica de la assertion');
  }

  // Sugerencias generales
  if (context.screenshots && context.screenshots.length > 0) {
    suggestions.push(`Revisar screenshot: ${context.screenshots[0]}`);
  }

  if (context.consoleErrors && context.consoleErrors.length > 0) {
    suggestions.push('Revisar errores de consola del navegador');
  }

  return {
    context,
    stackTrace: error.stack || 'No stack trace available',
    suggestions,
    severity,
  };
}

// ============================================================================
// Error Reporting
// ============================================================================

/**
 * Genera un reporte de error formateado
 */
export function formatErrorReport(report: ErrorReport): string {
  const { context, stackTrace, suggestions, severity } = report;
  const severityEmoji = {
    low: 'ℹ️',
    medium: '⚠️',
    high: '❌',
    critical: '🚨',
  };

  const lines: string[] = [];

  lines.push('='.repeat(60));
  lines.push(`${severityEmoji[severity]} Error Report - ${severity.toUpperCase()}`);
  lines.push('='.repeat(60));
  lines.push(`Test: ${context.testName}`);
  lines.push(`Step: ${context.step}`);
  lines.push(`Timestamp: ${new Date(context.timestamp).toISOString()}`);
  lines.push('');
  lines.push('Error:');
  lines.push(`  ${context.error.message}`);
  lines.push('');
  lines.push('Stack Trace:');
  lines.push(stackTrace.split('\n').map((line) => `  ${line}`).join('\n'));
  lines.push('');

  if (suggestions.length > 0) {
    lines.push('Sugerencias:');
    suggestions.forEach((suggestion, index) => {
      lines.push(`  ${index + 1}. ${suggestion}`);
    });
    lines.push('');
  }

  if (context.screenshots && context.screenshots.length > 0) {
    lines.push('Screenshots:');
    context.screenshots.forEach((screenshot) => {
      lines.push(`  - ${screenshot}`);
    });
    lines.push('');
  }

  if (context.consoleErrors && context.consoleErrors.length > 0) {
    lines.push('Console Errors:');
    context.consoleErrors.forEach((error) => {
      lines.push(`  - ${error}`);
    });
    lines.push('');
  }

  lines.push('='.repeat(60));

  return lines.join('\n');
}

/**
 * Maneja un error de forma consistente
 */
export async function handleTestError(
  testName: string,
  step: string,
  error: Error,
  page?: Page
): Promise<ErrorReport> {
  const context = await collectErrorContext(testName, step, error, page);
  const report = analyzeError(context);
  const formattedReport = formatErrorReport(report);

  console.error(formattedReport);

  return report;
}

// ============================================================================
// Error Recovery
// ============================================================================

/**
 * Intenta recuperarse de un error común
 */
export async function attemptErrorRecovery(
  page: Page,
  error: Error
): Promise<boolean> {
  const errorMessage = error.message.toLowerCase();

  // Recuperación para errores de timeout
  if (errorMessage.includes('timeout')) {
    try {
      await page.reload({ waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      return true;
    } catch (recoveryError) {
      return false;
    }
  }

  // Recuperación para errores de elemento no encontrado
  if (errorMessage.includes('not found') || errorMessage.includes('selector')) {
    try {
      await page.waitForTimeout(1000);
      return true;
    } catch (recoveryError) {
      return false;
    }
  }

  return false;
}



