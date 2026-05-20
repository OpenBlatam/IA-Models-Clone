// import "server-only";

import { cache } from "react";
import { getServerSession } from "next-auth";
import { authOptions } from "@/auth";

export const getCurrentUser = cache(async () => {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return undefined;
    }
    return session.user;
  } catch (error) {
    console.error("Error getting current user:", error);
    return undefined;
  }
});