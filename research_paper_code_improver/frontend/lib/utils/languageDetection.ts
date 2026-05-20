/**
 * Language detection utilities for code analysis
 */

export type SupportedLanguage =
  | 'python'
  | 'javascript'
  | 'typescript'
  | 'java'
  | 'cpp'
  | 'c'
  | 'go'
  | 'rust'
  | 'php'
  | 'ruby'

export const SUPPORTED_LANGUAGES: SupportedLanguage[] = [
  'python',
  'javascript',
  'typescript',
  'java',
  'cpp',
  'c',
  'go',
  'rust',
  'php',
  'ruby',
]

export const LANGUAGE_LABELS: Record<SupportedLanguage, string> = {
  python: 'Python',
  javascript: 'JavaScript',
  typescript: 'TypeScript',
  java: 'Java',
  cpp: 'C++',
  c: 'C',
  go: 'Go',
  rust: 'Rust',
  php: 'PHP',
  ruby: 'Ruby',
}

/**
 * Detects the programming language from code content
 * @param code - The code string to analyze
 * @returns Detected language or 'python' as default
 */
export function detectLanguage(code: string): SupportedLanguage {
  if (!code || !code.trim()) {
    return 'python' // default
  }

  const codeLower = code.toLowerCase()

  // Python detection
  if (
    code.includes('def ') ||
    code.includes('import ') ||
    code.includes('print(') ||
    code.includes('from ') ||
    code.includes('__init__') ||
    code.includes('if __name__')
  ) {
    return 'python'
  }

  // TypeScript detection (check before JavaScript)
  if (
    (code.includes('function ') || code.includes('const ') || code.includes('let ')) &&
    (code.includes(': ') || code.includes('interface ') || code.includes('type ') || code.includes('enum '))
  ) {
    return 'typescript'
  }

  // JavaScript detection
  if (
    code.includes('function ') ||
    code.includes('const ') ||
    code.includes('let ') ||
    code.includes('var ') ||
    code.includes('=>') ||
    code.includes('module.exports')
  ) {
    return 'javascript'
  }

  // Java detection
  if (
    code.includes('public class') ||
    code.includes('public static') ||
    code.includes('@Override') ||
    code.includes('package ') ||
    code.includes('import java.')
  ) {
    return 'java'
  }

  // C++ detection
  if (
    code.includes('#include') ||
    code.includes('std::') ||
    code.includes('namespace ') ||
    code.includes('using namespace')
  ) {
    return 'cpp'
  }

  // C detection
  if (
    code.includes('#include') &&
    !code.includes('std::') &&
    !code.includes('namespace') &&
    !code.includes('class ')
  ) {
    return 'c'
  }

  // Go detection
  if (
    code.includes('package main') ||
    code.includes('func ') ||
    code.includes(':= ') ||
    code.includes('import "')
  ) {
    return 'go'
  }

  // Rust detection
  if (
    code.includes('fn ') ||
    code.includes('let mut') ||
    code.includes('use ') ||
    code.includes('impl ') ||
    code.includes('struct ')
  ) {
    return 'rust'
  }

  // PHP detection
  if (
    code.includes('<?php') ||
    code.includes('<?=') ||
    code.includes('$') ||
    code.includes('->')
  ) {
    return 'php'
  }

  // Ruby detection
  if (
    code.includes('def ') && code.includes('end') ||
    code.includes('class ') && code.includes('end') ||
    code.includes('require ') ||
    code.includes('@')
  ) {
    return 'ruby'
  }

  // Default to Python
  return 'python'
}

/**
 * Gets language options for select components
 */
export function getLanguageOptions() {
  return SUPPORTED_LANGUAGES.map((lang) => ({
    value: lang,
    label: LANGUAGE_LABELS[lang],
  }))
}




