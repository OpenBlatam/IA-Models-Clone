/**
 * Enhanced formatting utilities
 * 
 * Provides additional formatting functions beyond the base formatting utilities
 */

/**
 * Formats a number as currency
 */
export function formatCurrency(
  amount: number,
  currency: string = "USD",
  locale: string = "es-ES"
): string {
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency,
  }).format(amount);
}

/**
 * Formats a number as percentage
 */
export function formatPercentage(
  value: number,
  decimals: number = 2,
  locale: string = "es-ES"
): string {
  return new Intl.NumberFormat(locale, {
    style: "percent",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value / 100);
}

/**
 * Formats a number with unit
 */
export function formatNumberWithUnit(
  value: number,
  unit: string,
  locale: string = "es-ES"
): string {
  return new Intl.NumberFormat(locale).format(value) + ` ${unit}`;
}

/**
 * Formats bytes to human-readable size
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

/**
 * Formats duration in seconds to human-readable string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  if (minutes < 60) {
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  if (hours < 24) {
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  }

  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;

  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
}

/**
 * Formats frequency in seconds to human-readable string
 */
export function formatFrequency(frequencySeconds: number): string {
  if (frequencySeconds < 60) {
    return `Cada ${frequencySeconds} segundos`;
  }

  const minutes = Math.floor(frequencySeconds / 60);
  if (minutes < 60) {
    return minutes === 1 ? "Cada minuto" : `Cada ${minutes} minutos`;
  }

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return hours === 1 ? "Cada hora" : `Cada ${hours} horas`;
  }

  const days = Math.floor(hours / 24);
  return days === 1 ? "Cada día" : `Cada ${days} días`;
}

/**
 * Formats a large number with abbreviations (K, M, B)
 */
export function formatCompactNumber(value: number, locale: string = "es-ES"): string {
  return new Intl.NumberFormat(locale, {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

/**
 * Formats a phone number
 */
export function formatPhoneNumber(phone: string, countryCode: string = "+34"): string {
  const cleaned = phone.replace(/\D/g, "");
  if (cleaned.length === 9) {
    return `${countryCode} ${cleaned.slice(0, 3)} ${cleaned.slice(3, 6)} ${cleaned.slice(6)}`;
  }
  return phone;
}

/**
 * Formats a credit card number (masks all but last 4 digits)
 */
export function formatCreditCard(cardNumber: string): string {
  const cleaned = cardNumber.replace(/\D/g, "");
  if (cleaned.length >= 4) {
    const last4 = cleaned.slice(-4);
    const masked = "*".repeat(cleaned.length - 4);
    return `${masked}${last4}`;
  }
  return cardNumber;
}

/**
 * Formats a file size with appropriate unit
 */
export function formatFileSize(bytes: number): string {
  return formatBytes(bytes);
}

/**
 * Formats execution time in milliseconds
 */
export function formatExecutionTime(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  const seconds = ms / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  }
  return formatDuration(Math.floor(seconds));
}




