/**
 * Barrel exports for Continuous Agent constants
 */

export {
  REFRESH_INTERVALS,
  TASK_TYPES,
  TASK_TYPE_LABELS,
  TASK_TYPE_OPTIONS,
  FORM_DEFAULTS,
  FREQUENCY_EXAMPLES,
} from "./config";
export { VALIDATION_LIMITS } from "../utils/validation/constants";
export { SUCCESS_MESSAGES, ERROR_MESSAGES, UI_MESSAGES } from "./messages";
export {
  PROMPT_TEMPLATES,
  getTemplatesByCategory,
  getTemplateByName,
  type PromptTemplate,
} from "./prompt-templates";
export {
  ANIMATION_DURATION,
  ANIMATION_EASING,
  cardAnimations,
  modalAnimations,
  fadeInAnimation,
} from "./animations";
export {
  CARD_STYLES,
  TEXT_STYLES,
  STATUS_STYLES,
  STATUS_TEXT,
  BADGE_STYLES,
  MODAL_STYLES,
  FORM_STYLES,
  LAYOUT_STYLES,
} from "./styles";
export { STATE_STYLES } from "./state-styles";
