"use client";

import { AblyProvider } from "ably/react";
import * as Ably from "ably";

const client = new Ably.Realtime({
  authUrl: "/api/ably-token",
  authMethod: "POST",
  clientId: "anonymous",
  recover: (lastConnectionDetails, cb) => {
    cb(true);
  },
  closeOnUnload: true,
});

export function AblySpacesProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AblyProvider client={client}>
      {children}
    </AblyProvider>
  );
} 