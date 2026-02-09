/**
 * Configuration export/import utilities
 * @module robot-3d-view/utils/config-export
 */

import type { SceneConfig } from '../schemas/validation-schemas';
import { sceneConfigSchema } from '../schemas/validation-schemas';

/**
 * Exports configuration to JSON file
 * 
 * @param config - Configuration to export
 * @param filename - Filename for export
 */
export function exportConfig(config: SceneConfig, filename = 'robot-3d-view-config.json'): void {
  const json = JSON.stringify(config, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Imports configuration from file
 * 
 * @param file - File to import
 * @returns Promise that resolves to imported configuration
 */
export async function importConfig(file: File): Promise<SceneConfig> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const parsed = JSON.parse(text);
        const validated = sceneConfigSchema.parse(parsed);
        resolve(validated);
      } catch (error) {
        reject(new Error('Invalid configuration file'));
      }
    };

    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };

    reader.readAsText(file);
  });
}

/**
 * Copies configuration to clipboard
 * 
 * @param config - Configuration to copy
 */
export async function copyConfigToClipboard(config: SceneConfig): Promise<void> {
  const json = JSON.stringify(config, null, 2);
  await navigator.clipboard.writeText(json);
}

/**
 * Gets configuration from clipboard
 * 
 * @returns Promise that resolves to configuration from clipboard
 */
export async function getConfigFromClipboard(): Promise<SceneConfig> {
  const text = await navigator.clipboard.readText();
  const parsed = JSON.parse(text);
  return sceneConfigSchema.parse(parsed);
}



