"use client";

import React from "react";
import { STATE_STYLES } from "../constants/state-styles";
import { UI_MESSAGES } from "../constants/messages";

/**
 * Loading state component for the agent page
 */
export const AgentLoadingState = (): JSX.Element => {
  return (
    <div className={STATE_STYLES.CONTAINER}>
      <div className={STATE_STYLES.LOADING} role="status" aria-live="polite" aria-busy="true">
        {UI_MESSAGES.LOADING_AGENTS}
      </div>
    </div>
  );
};

