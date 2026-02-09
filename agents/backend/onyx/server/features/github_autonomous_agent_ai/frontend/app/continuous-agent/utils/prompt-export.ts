/**
 * Utilities for exporting and importing prompts
 */

import type { PromptTemplate } from "../constants/prompt-templates";

/**
 * Exports a prompt to a downloadable file
 * @param prompt - The prompt to export
 * @param filename - Optional filename (default: "prompt.txt")
 */
export const exportPrompt = (prompt: string, filename: string = "prompt.txt"): void => {
  if (!prompt || !prompt.trim()) {
    throw new Error("Prompt is empty");
  }

  const blob = new Blob([prompt], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Exports a prompt template to JSON
 * @param template - The template to export
 * @param filename - Optional filename (default: "template.json")
 */
export const exportTemplate = (
  template: PromptTemplate,
  filename: string = "template.json"
): void => {
  const json = JSON.stringify(
    {
      name: template.name,
      description: template.description,
      category: template.category,
      value: template.value,
      exportedAt: new Date().toISOString(),
    },
    null,
    2
  );

  const blob = new Blob([json], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Imports a prompt from a file
 * @param file - The file to import
 * @returns Promise resolving to the prompt content
 */
export const importPrompt = async (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject(new Error("No file provided"));
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result;
      if (typeof content === "string") {
        resolve(content);
      } else {
        reject(new Error("Failed to read file"));
      }
    };
    reader.onerror = () => reject(new Error("Error reading file"));
    reader.readAsText(file);
  });
};

/**
 * Imports a template from a JSON file
 * @param file - The JSON file to import
 * @returns Promise resolving to the template
 */
export const importTemplate = async (file: File): Promise<PromptTemplate> => {
  const content = await importPrompt(file);
  try {
    const parsed = JSON.parse(content);
    
    if (!parsed.value || typeof parsed.value !== "string") {
      throw new Error("Invalid template format: missing 'value' field");
    }

    return {
      name: parsed.name || "Imported Template",
      description: parsed.description || "",
      category: parsed.category || "custom",
      value: parsed.value,
    };
  } catch (error) {
    throw new Error(`Invalid JSON format: ${error instanceof Error ? error.message : "Unknown error"}`);
  }
};

/**
 * Copies prompt to clipboard
 * @param prompt - The prompt to copy
 * @returns Promise that resolves when copied
 */
export const copyPromptToClipboard = async (prompt: string): Promise<void> => {
  if (!prompt || !prompt.trim()) {
    throw new Error("Prompt is empty");
  }

  try {
    await navigator.clipboard.writeText(prompt);
  } catch (error) {
    // Fallback for older browsers
    const textArea = document.createElement("textarea");
    textArea.value = prompt;
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);
  }
};




