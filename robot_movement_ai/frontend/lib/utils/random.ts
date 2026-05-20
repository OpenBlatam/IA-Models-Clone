/**
 * Random utilities
 */

// Random integer between min and max (inclusive)
export function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Random float between min and max
export function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

// Random boolean
export function randomBoolean(): boolean {
  return Math.random() >= 0.5;
}

// Random item from array
export function randomItem<T>(array: T[]): T | undefined {
  if (array.length === 0) return undefined;
  return array[randomInt(0, array.length - 1)];
}

// Random items from array (without duplicates)
export function randomItems<T>(array: T[], count: number): T[] {
  const shuffled = [...array].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(count, array.length));
}

// Random string
export function randomString(length: number = 8, charset: string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'): string {
  let result = '';
  for (let i = 0; i < length; i++) {
    result += charset.charAt(Math.floor(Math.random() * charset.length));
  }
  return result;
}

// Random UUID (v4)
export function randomUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Random color (hex)
export function randomColor(): string {
  return `#${Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0')}`;
}

// Random color (RGB)
export function randomColorRGB(): { r: number; g: number; b: number } {
  return {
    r: randomInt(0, 255),
    g: randomInt(0, 255),
    b: randomInt(0, 255),
  };
}

// Shuffle array (Fisher-Yates)
export function shuffle<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// Weighted random
export function weightedRandom<T>(items: Array<{ item: T; weight: number }>): T | undefined {
  const totalWeight = items.reduce((sum, item) => sum + item.weight, 0);
  let random = Math.random() * totalWeight;

  for (const { item, weight } of items) {
    random -= weight;
    if (random <= 0) {
      return item;
    }
  }

  return items[items.length - 1]?.item;
}



