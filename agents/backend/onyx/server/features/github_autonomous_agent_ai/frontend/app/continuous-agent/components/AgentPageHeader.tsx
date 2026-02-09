"use client";

import React from "react";

/**
 * Header component for the Continuous Agent page
 */
export const AgentPageHeader = (): JSX.Element => {
  return (
    <header className="mb-8">
      <h1 className="text-3xl font-bold mb-2">Agente Continuo 24/7</h1>
      <p className="text-muted-foreground">
        Crea agentes que funcionan las 24 horas, incluso cuando tu computadora está apagada.
        Se detienen automáticamente cuando se acaban los créditos de Stripe o cuando los desactives.
      </p>
    </header>
  );
};



