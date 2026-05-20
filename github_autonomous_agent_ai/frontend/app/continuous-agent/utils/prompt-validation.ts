/**
 * Advanced validation utilities for Perplexity-style prompts
 * 
 * Provides validation for prompt structure, tags, and content
 */

export interface PromptValidationResult {
  readonly isValid: boolean;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
  readonly stats: {
    readonly hasGoalTag: boolean;
    readonly hasFormatRules: boolean;
    readonly hasRestrictions: boolean;
    readonly hasQueryType: boolean;
    readonly hasPlanningRules: boolean;
    readonly hasOutput: boolean;
    readonly tagCount: number;
    readonly characterCount: number;
    readonly lineCount: number;
  };
}

/**
 * Validates Perplexity-style prompt structure
 * @param prompt - The prompt to validate
 * @returns Validation result with errors, warnings, and stats
 */
export const validatePromptStructure = (prompt: string): PromptValidationResult => {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!prompt || !prompt.trim()) {
    return {
      isValid: true,
      errors: [],
      warnings: [],
      stats: {
        hasGoalTag: false,
        hasFormatRules: false,
        hasRestrictions: false,
        hasQueryType: false,
        hasPlanningRules: false,
        hasOutput: false,
        tagCount: 0,
        characterCount: 0,
        lineCount: 0,
      },
    };
  }

  const trimmed = prompt.trim();
  const lines = trimmed.split("\n");
  const characterCount = trimmed.length;
  const lineCount = lines.length;

  // Check for common Perplexity-style tags
  const hasGoalTag = /<goal>/i.test(trimmed) && /<\/goal>/i.test(trimmed);
  const hasFormatRules = /<format_rules>/i.test(trimmed) && /<\/format_rules>/i.test(trimmed);
  const hasRestrictions = /<restrictions>/i.test(trimmed) && /<\/restrictions>/i.test(trimmed);
  const hasQueryType = /<query_type>/i.test(trimmed) && /<\/query_type>/i.test(trimmed);
  const hasPlanningRules = /<planning_rules>/i.test(trimmed) && /<\/planning_rules>/i.test(trimmed);
  const hasOutput = /<output>/i.test(trimmed) && /<\/output>/i.test(trimmed);

  // Count XML-like tags
  const tagMatches = trimmed.match(/<[^>]+>/g);
  const tagCount = tagMatches ? tagMatches.length : 0;

  // Validate tag pairs
  if (hasGoalTag) {
    const goalOpen = (trimmed.match(/<goal>/gi) || []).length;
    const goalClose = (trimmed.match(/<\/goal>/gi) || []).length;
    if (goalOpen !== goalClose) {
      errors.push("Tag <goal> no está balanceado (abre/cierra)");
    }
  }

  if (hasFormatRules) {
    const formatOpen = (trimmed.match(/<format_rules>/gi) || []).length;
    const formatClose = (trimmed.match(/<\/format_rules>/gi) || []).length;
    if (formatOpen !== formatClose) {
      errors.push("Tag <format_rules> no está balanceado");
    }
  }

  if (hasRestrictions) {
    const restrictionsOpen = (trimmed.match(/<restrictions>/gi) || []).length;
    const restrictionsClose = (trimmed.match(/<\/restrictions>/gi) || []).length;
    if (restrictionsOpen !== restrictionsClose) {
      errors.push("Tag <restrictions> no está balanceado");
    }
  }

  // Warnings for missing recommended tags
  if (!hasGoalTag) {
    warnings.push("No se encontró tag <goal>. Es recomendado para prompts estilo Perplexity.");
  }

  if (!hasFormatRules) {
    warnings.push("No se encontró tag <format_rules>. Es recomendado para definir formato de respuesta.");
  }

  // Check for very long prompts
  if (characterCount > 8000) {
    warnings.push("El prompt es muy largo (>8000 caracteres). Considera simplificarlo.");
  }

  // Check for very short prompts
  if (characterCount < 50 && hasGoalTag) {
    warnings.push("El prompt es muy corto. Asegúrate de incluir instrucciones suficientes.");
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    stats: {
      hasGoalTag,
      hasFormatRules,
      hasRestrictions,
      hasQueryType,
      hasPlanningRules,
      hasOutput,
      tagCount,
      characterCount,
      lineCount,
    },
  };
};

/**
 * Extracts goal content from prompt
 * @param prompt - The prompt to extract from
 * @returns Goal content or null if not found
 */
export const extractGoalContent = (prompt: string): string | null => {
  const match = prompt.match(/<goal>(.*?)<\/goal>/is);
  return match ? match[1].trim() : null;
};

/**
 * Extracts format rules from prompt
 * @param prompt - The prompt to extract from
 * @returns Format rules content or null if not found
 */
export const extractFormatRules = (prompt: string): string | null => {
  const match = prompt.match(/<format_rules>(.*?)<\/format_rules>/is);
  return match ? match[1].trim() : null;
};

/**
 * Checks if prompt follows Perplexity-style structure
 * @param prompt - The prompt to check
 * @returns True if it appears to be a Perplexity-style prompt
 */
export const isPerplexityStylePrompt = (prompt: string): boolean => {
  if (!prompt || !prompt.trim()) {
    return false;
  }

  const validation = validatePromptStructure(prompt);
  return validation.stats.hasGoalTag && validation.stats.hasFormatRules;
};

/**
 * Gets prompt structure summary
 * @param prompt - The prompt to analyze
 * @returns Summary string
 */
export const getPromptSummary = (prompt: string): string => {
  if (!prompt || !prompt.trim()) {
    return "Prompt vacío";
  }

  const validation = validatePromptStructure(prompt);
  const { stats } = validation;

  const parts: string[] = [];
  if (stats.hasGoalTag) parts.push("Goal");
  if (stats.hasFormatRules) parts.push("Format Rules");
  if (stats.hasRestrictions) parts.push("Restrictions");
  if (stats.hasQueryType) parts.push("Query Type");
  if (stats.hasPlanningRules) parts.push("Planning Rules");
  if (stats.hasOutput) parts.push("Output");

  const structure = parts.length > 0 ? parts.join(", ") : "Estructura básica";
  return `${structure} • ${stats.characterCount} caracteres • ${stats.lineCount} líneas`;
};




