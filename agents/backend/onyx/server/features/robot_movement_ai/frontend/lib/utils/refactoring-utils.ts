/**
 * Utilities for refactoring and code migration
 */

/**
 * Replace console.log with logger
 */
export function replaceConsoleLog(code: string): string {
  return code
    .replace(/console\.log\(/g, 'logger.info(')
    .replace(/console\.error\(/g, 'logger.error(')
    .replace(/console\.warn\(/g, 'logger.warn(')
    .replace(/console\.debug\(/g, 'logger.debug(');
}

/**
 * Extract inline styles to constants
 */
export function extractStyles(styles: Record<string, any>): string {
  return Object.entries(styles)
    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
    .join(',\n  ');
}

/**
 * Generate hook from component logic
 */
export function generateHookFromComponent(
  componentName: string,
  stateVariables: string[],
  effects: string[]
): string {
  return `
export function use${componentName}() {
  ${stateVariables.map((v) => `const [${v}, set${v.charAt(0).toUpperCase() + v.slice(1)}] = useState();`).join('\n  ')}

  ${effects.map((e) => `useEffect(() => { ${e} }, []);`).join('\n  ')}

  return {
    ${stateVariables.map((v) => `${v},`).join('\n    ')}
  };
}
  `.trim();
}



