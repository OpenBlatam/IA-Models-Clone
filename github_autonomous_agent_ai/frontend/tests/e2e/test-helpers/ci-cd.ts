/**
 * Utilidades para CI/CD Integration
 * 
 * Proporciona funciones para integrar tests con sistemas de CI/CD
 */
import { TestInfo } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

export interface CICDReport {
  testName: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
  error?: string;
  screenshot?: string;
  video?: string;
  trace?: string;
  metadata?: Record<string, any>;
}

export interface CICDSummary {
  total: number;
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  timestamp: string;
}

// ============================================================================
// CI/CD Detection
// ============================================================================

/**
 * Detecta si estamos ejecutando en CI/CD
 */
export function isCI(): boolean {
  return (
    process.env.CI === 'true' ||
    process.env.CONTINUOUS_INTEGRATION === 'true' ||
    !!process.env.GITHUB_ACTIONS ||
    !!process.env.GITLAB_CI ||
    !!process.env.JENKINS_URL ||
    !!process.env.TEAMCITY_VERSION ||
    !!process.env.BUILDKITE ||
    !!process.env.CIRCLECI ||
    !!process.env.TRAVIS
  );
}

/**
 * Obtiene información del entorno CI/CD
 */
export function getCIEnvironment(): {
  provider: string;
  buildId?: string;
  buildUrl?: string;
  commit?: string;
  branch?: string;
} {
  if (process.env.GITHUB_ACTIONS) {
    return {
      provider: 'GitHub Actions',
      buildId: process.env.GITHUB_RUN_ID,
      buildUrl: `${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`,
      commit: process.env.GITHUB_SHA,
      branch: process.env.GITHUB_REF?.replace('refs/heads/', ''),
    };
  }

  if (process.env.GITLAB_CI) {
    return {
      provider: 'GitLab CI',
      buildId: process.env.CI_PIPELINE_ID,
      buildUrl: process.env.CI_PIPELINE_URL,
      commit: process.env.CI_COMMIT_SHA,
      branch: process.env.CI_COMMIT_REF_NAME,
    };
  }

  if (process.env.JENKINS_URL) {
    return {
      provider: 'Jenkins',
      buildId: process.env.BUILD_NUMBER,
      buildUrl: process.env.BUILD_URL,
      commit: process.env.GIT_COMMIT,
      branch: process.env.GIT_BRANCH,
    };
  }

  return {
    provider: 'Unknown',
  };
}

// ============================================================================
// JUnit XML Report
// ============================================================================

/**
 * Genera reporte en formato JUnit XML para CI/CD
 */
export function generateJUnitXML(
  reports: CICDReport[],
  suiteName: string = 'E2E Tests'
): string {
  const summary = calculateSummary(reports);
  const timestamp = new Date().toISOString();

  let xml = `<?xml version="1.0" encoding="UTF-8"?>\n`;
  xml += `<testsuites name="${suiteName}" tests="${summary.total}" failures="${summary.failed}" skipped="${summary.skipped}" time="${summary.duration / 1000}" timestamp="${timestamp}">\n`;
  xml += `  <testsuite name="${suiteName}" tests="${summary.total}" failures="${summary.failed}" skipped="${summary.skipped}" time="${summary.duration / 1000}" timestamp="${timestamp}">\n`;

  for (const report of reports) {
    const status = report.status === 'failed' ? 'failure' : report.status;
    xml += `    <testcase name="${escapeXML(report.testName)}" time="${report.duration / 1000}"`;
    
    if (report.status === 'skipped') {
      xml += `>\n      <skipped/>\n    </testcase>\n`;
    } else if (report.status === 'failed') {
      xml += `>\n      <failure message="${escapeXML(report.error || 'Test failed')}">${escapeXML(report.error || '')}</failure>\n    </testcase>\n`;
    } else {
      xml += `/>\n`;
    }
  }

  xml += `  </testsuite>\n`;
  xml += `</testsuites>`;

  return xml;
}

// ============================================================================
// JSON Report
// ============================================================================

/**
 * Genera reporte en formato JSON para CI/CD
 */
export function generateJSONReport(
  reports: CICDReport[],
  metadata?: Record<string, any>
): string {
  const summary = calculateSummary(reports);
  const ciEnv = getCIEnvironment();

  const report = {
    summary,
    environment: ciEnv,
    tests: reports,
    metadata: metadata || {},
    timestamp: new Date().toISOString(),
  };

  return JSON.stringify(report, null, 2);
}

// ============================================================================
// GitHub Actions Annotations
// ============================================================================

/**
 * Crea una anotación para GitHub Actions
 */
export function createGitHubAnnotation(
  type: 'notice' | 'warning' | 'error',
  message: string,
  file?: string,
  line?: number,
  col?: number
): string {
  const annotation = `::${type}`;
  const params: string[] = [];

  if (file) {
    params.push(`file=${file}`);
  }
  if (line !== undefined) {
    params.push(`line=${line}`);
  }
  if (col !== undefined) {
    params.push(`col=${col}`);
  }

  const paramsStr = params.length > 0 ? ` ${params.join(',')}` : '';
  return `${annotation}${paramsStr}::${message}`;
}

/**
 * Crea un resumen para GitHub Actions
 */
export function createGitHubSummary(
  reports: CICDReport[]
): string {
  const summary = calculateSummary(reports);
  const passRate = summary.total > 0 
    ? ((summary.passed / summary.total) * 100).toFixed(2) 
    : '0.00';

  let markdown = `# Test Results\n\n`;
  markdown += `## Summary\n\n`;
  markdown += `- **Total**: ${summary.total}\n`;
  markdown += `- **Passed**: ${summary.passed} ✅\n`;
  markdown += `- **Failed**: ${summary.failed} ❌\n`;
  markdown += `- **Skipped**: ${summary.skipped} ⏭️\n`;
  markdown += `- **Pass Rate**: ${passRate}%\n`;
  markdown += `- **Duration**: ${(summary.duration / 1000).toFixed(2)}s\n\n`;

  if (summary.failed > 0) {
    markdown += `## Failed Tests\n\n`;
    for (const report of reports.filter((r) => r.status === 'failed')) {
      markdown += `### ${report.testName}\n\n`;
      markdown += `\`\`\`\n${report.error || 'No error message'}\n\`\`\`\n\n`;
    }
  }

  return markdown;
}

// ============================================================================
// Slack Notifications
// ============================================================================

/**
 * Genera payload para notificación de Slack
 */
export function createSlackPayload(
  reports: CICDReport[],
  webhookUrl?: string
): {
  text: string;
  blocks: Array<Record<string, any>>;
} {
  const summary = calculateSummary(reports);
  const ciEnv = getCIEnvironment();
  const passRate = summary.total > 0 
    ? ((summary.passed / summary.total) * 100).toFixed(2) 
    : '0.00';

  const color = summary.failed === 0 ? 'good' : 'danger';
  const emoji = summary.failed === 0 ? '✅' : '❌';

  return {
    text: `${emoji} Test Results: ${summary.passed}/${summary.total} passed`,
    blocks: [
      {
        type: 'header',
        text: {
          type: 'plain_text',
          text: `${emoji} E2E Test Results`,
        },
      },
      {
        type: 'section',
        fields: [
          {
            type: 'mrkdwn',
            text: `*Total:* ${summary.total}`,
          },
          {
            type: 'mrkdwn',
            text: `*Passed:* ${summary.passed} ✅`,
          },
          {
            type: 'mrkdwn',
            text: `*Failed:* ${summary.failed} ❌`,
          },
          {
            type: 'mrkdwn',
            text: `*Pass Rate:* ${passRate}%`,
          },
        ],
      },
      {
        type: 'section',
        fields: [
          {
            type: 'mrkdwn',
            text: `*Duration:* ${(summary.duration / 1000).toFixed(2)}s`,
          },
          {
            type: 'mrkdwn',
            text: `*Provider:* ${ciEnv.provider}`,
          },
        ],
      },
      {
        type: 'divider',
      },
    ],
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function calculateSummary(reports: CICDReport[]): CICDSummary {
  return {
    total: reports.length,
    passed: reports.filter((r) => r.status === 'passed').length,
    failed: reports.filter((r) => r.status === 'failed').length,
    skipped: reports.filter((r) => r.status === 'skipped').length,
    duration: reports.reduce((sum, r) => sum + r.duration, 0),
    timestamp: new Date().toISOString(),
  };
}

function escapeXML(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * Guarda reporte en archivo
 */
export async function saveCICDReport(
  report: string,
  format: 'xml' | 'json' | 'md',
  outputDir: string = 'test-results'
): Promise<string> {
  const extension = format === 'xml' ? 'xml' : format === 'json' ? 'json' : 'md';
  const filename = `test-results-${Date.now()}.${extension}`;
  const filepath = path.join(outputDir, filename);

  // Crear directorio si no existe
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  fs.writeFileSync(filepath, report, 'utf-8');
  return filepath;
}

/**
 * Crea reporte desde TestInfo de Playwright
 */
export function createCICDReportFromTestInfo(
  testInfo: TestInfo,
  metadata?: Record<string, any>
): CICDReport {
  const status =
    testInfo.status === 'passed'
      ? 'passed'
      : testInfo.status === 'skipped'
      ? 'skipped'
      : 'failed';

  return {
    testName: testInfo.title,
    status,
    duration: testInfo.duration,
    error: testInfo.error?.message,
    screenshot: testInfo.attachments.find((a) => a.name === 'screenshot')?.path,
    video: testInfo.attachments.find((a) => a.name === 'video')?.path,
    trace: testInfo.attachments.find((a) => a.name === 'trace')?.path,
    metadata,
  };
}


