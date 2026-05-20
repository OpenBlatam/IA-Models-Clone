/**
 * Analytics utilities for prompts
 * 
 * Provides insights and statistics about agent prompts
 */

import type { ContinuousAgent } from "../types";
import { validatePromptStructure, isPerplexityStylePrompt } from "./prompt-validation";
import { countPromptTags } from "./prompt-highlighting";

export interface PromptAnalytics {
  readonly totalAgents: number;
  readonly agentsWithPrompts: number;
  readonly agentsWithoutPrompts: number;
  readonly perplexityStylePrompts: number;
  readonly averagePromptLength: number;
  readonly totalPromptLength: number;
  readonly longestPrompt: number;
  readonly shortestPrompt: number;
  readonly validPrompts: number;
  readonly invalidPrompts: number;
  readonly promptsWithErrors: number;
  readonly promptsWithWarnings: number;
  readonly tagDistribution: Record<string, number>;
  readonly categoryDistribution: Record<string, number>;
}

/**
 * Analyzes prompts across all agents
 * @param agents - Array of agents to analyze
 * @returns Analytics object with statistics
 */
export const analyzePrompts = (agents: ContinuousAgent[]): PromptAnalytics => {
  const agentsWithPrompts = agents.filter((agent) => agent.config.goal?.trim());
  const agentsWithoutPrompts = agents.filter((agent) => !agent.config.goal?.trim());
  
  const promptLengths = agentsWithPrompts
    .map((agent) => agent.config.goal?.length || 0)
    .filter((length) => length > 0);

  const totalPromptLength = promptLengths.reduce((sum, len) => sum + len, 0);
  const averagePromptLength = promptLengths.length > 0
    ? Math.round(totalPromptLength / promptLengths.length)
    : 0;

  const longestPrompt = promptLengths.length > 0 ? Math.max(...promptLengths) : 0;
  const shortestPrompt = promptLengths.length > 0 ? Math.min(...promptLengths) : 0;

  const perplexityStylePrompts = agentsWithPrompts.filter((agent) =>
    isPerplexityStylePrompt(agent.config.goal || "")
  ).length;

  let validPrompts = 0;
  let invalidPrompts = 0;
  let promptsWithErrors = 0;
  let promptsWithWarnings = 0;

  const tagCounts: Record<string, number> = {};
  const categoryCounts: Record<string, number> = {};

  agentsWithPrompts.forEach((agent) => {
    const validation = validatePromptStructure(agent.config.goal || "");
    
    if (validation.isValid && validation.errors.length === 0) {
      validPrompts++;
    } else {
      invalidPrompts++;
    }

    if (validation.errors.length > 0) {
      promptsWithErrors++;
    }

    if (validation.warnings.length > 0) {
      promptsWithWarnings++;
    }

    // Count tags
    const tags = countPromptTags(agent.config.goal || "");
    Object.entries(tags).forEach(([tag, count]) => {
      tagCounts[tag] = (tagCounts[tag] || 0) + count;
    });

    // Count by task type (as category proxy)
    const taskType = agent.config.taskType;
    categoryCounts[taskType] = (categoryCounts[taskType] || 0) + 1;
  });

  return {
    totalAgents: agents.length,
    agentsWithPrompts: agentsWithPrompts.length,
    agentsWithoutPrompts: agentsWithoutPrompts.length,
    perplexityStylePrompts,
    averagePromptLength,
    totalPromptLength,
    longestPrompt,
    shortestPrompt,
    validPrompts,
    invalidPrompts,
    promptsWithErrors,
    promptsWithWarnings,
    tagDistribution: tagCounts,
    categoryDistribution: categoryCounts,
  };
};

/**
 * Gets prompt health score (0-100)
 * @param agents - Array of agents
 * @returns Health score
 */
export const getPromptHealthScore = (agents: ContinuousAgent[]): number => {
  const analytics = analyzePrompts(agents);
  
  if (analytics.agentsWithPrompts === 0) {
    return 0; // No prompts = 0 score
  }

  const validityScore = (analytics.validPrompts / analytics.agentsWithPrompts) * 50;
  const structureScore = (analytics.perplexityStylePrompts / analytics.agentsWithPrompts) * 30;
  const errorPenalty = (analytics.promptsWithErrors / analytics.agentsWithPrompts) * 20;

  return Math.round(Math.max(0, Math.min(100, validityScore + structureScore - errorPenalty)));
};




