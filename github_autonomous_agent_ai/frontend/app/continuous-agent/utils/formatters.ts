/**
 * Formatters for Continuous Agent feature
 */

export const formatFrequency = (seconds: number): string => {
  if (seconds < 60) {
    return `${seconds} seg`;
  }
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    return `${minutes} min${minutes !== 1 ? "" : ""}`;
  }
  if (seconds < 86400) {
    const hours = Math.floor(seconds / 3600);
    return `${hours} hora${hours !== 1 ? "s" : ""}`;
  }
  const days = Math.floor(seconds / 86400);
  const remainingHours = Math.floor((seconds % 86400) / 3600);
  if (remainingHours > 0) {
    return `${days} día${days !== 1 ? "s" : ""} y ${remainingHours} hora${remainingHours !== 1 ? "s" : ""}`;
  }
  return `${days} día${days !== 1 ? "s" : ""}`;
};

export const formatJSONError = (error: unknown): string => {
  if (error instanceof SyntaxError) {
    return `Error de sintaxis JSON: ${error.message}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Error al procesar JSON";
};

export const getJSONErrorPosition = (jsonString: string): number | null => {
  try {
    JSON.parse(jsonString);
    return null;
  } catch (error) {
    if (error instanceof SyntaxError && error.message.includes("position")) {
      const match = error.message.match(/position (\d+)/);
      if (match) {
        return parseInt(match[1], 10);
      }
    }
    return null;
  }
};



