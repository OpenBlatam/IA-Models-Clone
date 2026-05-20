"use client";

import React from "react";
import { STATE_STYLES } from "../constants/state-styles";

type AgentErrorStateProps = {
  readonly error: string;
};

/**
 * Error state component for the agent page
 */
export const AgentErrorState = ({ error }: AgentErrorStateProps): JSX.Element => {
  return (
    <div className={STATE_STYLES.CONTAINER}>
      <div className={STATE_STYLES.ERROR} role="alert" aria-live="assertive">
        {error}
      </div>
    </div>
  );
};

