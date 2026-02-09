/**
 * Chat Utils - Utilidades para Chat
 * ==================================
 * 
 * Funciones utilitarias para el sistema de chat
 */

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

/**
 * Formatear timestamp
 */
export const formatTimestamp = (date: Date, format: 'short' | 'long' | 'relative' = 'short'): string => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (format === 'relative') {
    if (seconds < 60) return 'Ahora';
    if (minutes < 60) return `Hace ${minutes} min`;
    if (hours < 24) return `Hace ${hours} h`;
    if (days < 7) return `Hace ${days} días`;
    return date.toLocaleDateString();
  }

  if (format === 'short') {
    return date.toLocaleTimeString('es-ES', {
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  return date.toLocaleString('es-ES', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

/**
 * Exportar conversación a varios formatos
 */
export const exportConversation = (
  messages: Message[],
  format: 'txt' | 'json' | 'markdown' | 'csv' = 'txt'
): string => {
  switch (format) {
    case 'json':
      return JSON.stringify(messages, null, 2);

    case 'markdown':
      return messages
        .map((msg) => {
          const role = msg.role === 'user' ? '**Usuario**' : '**Asistente**';
          const time = formatTimestamp(msg.timestamp, 'long');
          return `${role} (${time}):\n\n${msg.content}\n\n---\n`;
        })
        .join('\n');

    case 'csv':
      const headers = 'Role,Timestamp,Content\n';
      const rows = messages
        .map((msg) => {
          const role = msg.role;
          const time = msg.timestamp.toISOString();
          const content = `"${msg.content.replace(/"/g, '""')}"`;
          return `${role},${time},${content}`;
        })
        .join('\n');
      return headers + rows;

    case 'txt':
    default:
      return messages
        .map((msg) => {
          const role = msg.role === 'user' ? 'Usuario' : 'Asistente';
          const time = formatTimestamp(msg.timestamp, 'long');
          return `[${time}] ${role}: ${msg.content}`;
        })
        .join('\n\n');
  }
};

/**
 * Descargar archivo
 */
export const downloadFile = (content: string, filename: string, mimeType: string = 'text/plain'): void => {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

/**
 * Buscar en mensajes
 */
export const searchMessages = (messages: Message[], query: string): Message[] => {
  if (!query.trim()) return messages;
  
  const lowerQuery = query.toLowerCase();
  return messages.filter(
    (msg) =>
      msg.content.toLowerCase().includes(lowerQuery) ||
      msg.role.toLowerCase().includes(lowerQuery)
  );
};

/**
 * Generar título de conversación
 */
export const generateConversationTitle = (messages: Message[]): string => {
  if (messages.length === 0) return 'Nueva conversación';
  
  const firstUserMessage = messages.find((msg) => msg.role === 'user');
  if (!firstUserMessage) return 'Conversación sin título';
  
  const content = firstUserMessage.content.trim();
  if (content.length <= 50) return content;
  
  return content.substring(0, 47) + '...';
};

/**
 * Validar mensaje
 */
export const validateMessage = (content: string): { valid: boolean; error?: string } => {
  if (!content.trim()) {
    return { valid: false, error: 'El mensaje no puede estar vacío' };
  }
  
  if (content.length > 100000) {
    return { valid: false, error: 'El mensaje es demasiado largo (máximo 100,000 caracteres)' };
  }
  
  return { valid: true };
};

/**
 * Truncar mensaje
 */
export const truncateMessage = (content: string, maxLength: number = 100): string => {
  if (content.length <= maxLength) return content;
  return content.substring(0, maxLength - 3) + '...';
};

/**
 * Contar palabras
 */
export const countWords = (text: string): number => {
  return text.trim().split(/\s+/).filter((word) => word.length > 0).length;
};

/**
 * Contar caracteres
 */
export const countCharacters = (text: string): { total: number; withoutSpaces: number } => {
  return {
    total: text.length,
    withoutSpaces: text.replace(/\s/g, '').length,
  };
};

/**
 * Detectar idioma (simple)
 */
export const detectLanguage = (text: string): string => {
  const spanishWords = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo'];
  const englishWords = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for'];
  
  const lowerText = text.toLowerCase();
  const spanishCount = spanishWords.filter((word) => lowerText.includes(word)).length;
  const englishCount = englishWords.filter((word) => lowerText.includes(word)).length;
  
  if (spanishCount > englishCount) return 'es';
  if (englishCount > spanishCount) return 'en';
  return 'unknown';
};

/**
 * Formatear número
 */
export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat('es-ES').format(num);
};

/**
 * Debounce
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

/**
 * Throttle
 */
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};


