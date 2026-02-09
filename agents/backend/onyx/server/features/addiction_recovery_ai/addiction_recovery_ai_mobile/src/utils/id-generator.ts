import { nanoid } from 'nanoid';

export function generateId(size?: number): string {
  return nanoid(size);
}

export function generateShortId(): string {
  return nanoid(8);
}

