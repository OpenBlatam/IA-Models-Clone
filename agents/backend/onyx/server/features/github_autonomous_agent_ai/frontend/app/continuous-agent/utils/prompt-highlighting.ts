/**
 * Utilities for highlighting prompt syntax
 * 
 * Provides basic syntax highlighting for Perplexity-style prompts
 */

/**
 * Highlights XML-like tags in prompt text
 * @param text - The prompt text to highlight
 * @returns HTML string with highlighted tags
 */
export const highlightPromptTags = (text: string): string => {
  if (!text || !text.trim()) {
    return "";
  }

  // Escape HTML first
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Highlight opening tags
  const withOpeningTags = escaped.replace(
    /&lt;(\/?)(goal|format_rules|restrictions|query_type|planning_rules|output|personalization)(\s|&gt;)/gi,
    '<span class="text-blue-600 font-semibold">&lt;$1$2$3</span>'
  );

  // Highlight closing tags
  const withClosingTags = withOpeningTags.replace(
    /&lt;(\/)(goal|format_rules|restrictions|query_type|planning_rules|output|personalization)&gt;/gi,
    '<span class="text-blue-600 font-semibold">&lt;$1$2&gt;</span>'
  );

  return withClosingTags;
};

/**
 * Extracts and highlights specific sections
 * @param text - The prompt text
 * @param section - The section to highlight (e.g., "goal", "format_rules")
 * @returns Highlighted section or null
 */
export const highlightPromptSection = (
  text: string,
  section: "goal" | "format_rules" | "restrictions" | "query_type" | "planning_rules" | "output"
): string | null => {
  const regex = new RegExp(`<${section}>(.*?)</${section}>`, "is");
  const match = text.match(regex);
  
  if (!match) {
    return null;
  }

  return highlightPromptTags(match[1].trim());
};

/**
 * Gets color class for a tag type
 * @param tagName - The tag name
 * @returns Tailwind CSS class for the tag
 */
export const getTagColorClass = (tagName: string): string => {
  const tagColors: Record<string, string> = {
    goal: "text-blue-600",
    format_rules: "text-green-600",
    restrictions: "text-red-600",
    query_type: "text-purple-600",
    planning_rules: "text-orange-600",
    output: "text-indigo-600",
    personalization: "text-gray-600",
  };

  return tagColors[tagName.toLowerCase()] || "text-gray-600";
};

/**
 * Counts occurrences of each tag type
 * @param text - The prompt text
 * @returns Object with tag counts
 */
export const countPromptTags = (
  text: string
): Record<string, number> => {
  const tags = ["goal", "format_rules", "restrictions", "query_type", "planning_rules", "output", "personalization"];
  const counts: Record<string, number> = {};

  tags.forEach((tag) => {
    const regex = new RegExp(`<${tag}`, "gi");
    const matches = text.match(regex);
    counts[tag] = matches ? matches.length : 0;
  });

  return counts;
};




