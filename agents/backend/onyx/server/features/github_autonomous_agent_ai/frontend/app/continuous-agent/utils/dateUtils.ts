/**
 * Date formatting utilities for Continuous Agent
 * Uses native Intl APIs and date-fns for better performance and smaller bundle size
 */
import { formatDistanceToNow } from "date-fns";
import { es } from "date-fns/locale";

type DateInput = Date | string | null | undefined;

const parseDate = (dateInput: DateInput): Date | null => {
  if (!dateInput) {
    return null;
  }

  if (dateInput instanceof Date) {
    return dateInput;
  }

  const parsed = new Date(dateInput);
  return isNaN(parsed.getTime()) ? null : parsed;
};

export const formatRelativeTime = (dateInput: DateInput): string => {
  const date = parseDate(dateInput);
  if (!date) {
    return "Nunca";
  }

  try {
    return formatDistanceToNow(date, {
      addSuffix: true,
      locale: es,
    });
  } catch {
    return "Fecha inválida";
  }
};

export const formatDateTime = (dateInput: DateInput): string => {
  const date = parseDate(dateInput);
  if (!date) {
    return "No programado";
  }

  try {
    return new Intl.DateTimeFormat("es-ES", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  } catch {
    return "Fecha inválida";
  }
};

export const formatDate = (dateInput: DateInput): string => {
  const date = parseDate(dateInput);
  if (!date) {
    return "No disponible";
  }

  try {
    return new Intl.DateTimeFormat("es-ES", {
      year: "numeric",
      month: "long",
      day: "numeric",
    }).format(date);
  } catch {
    return "Fecha inválida";
  }
};

export const formatTime = (dateInput: DateInput): string => {
  const date = parseDate(dateInput);
  if (!date) {
    return "N/A";
  }

  try {
    return new Intl.DateTimeFormat("es-ES", {
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  } catch {
    return "Hora inválida";
  }
};



