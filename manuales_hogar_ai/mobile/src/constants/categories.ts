/**
 * Category Constants
 * =================
 * Category definitions matching backend categories
 */

export interface CategoryConfig {
  id: string;
  name: string;
  displayName: string;
  icon: string;
  description: string;
  color: string;
}

export const CATEGORIES: Record<string, CategoryConfig> = {
  plomeria: {
    id: 'plomeria',
    name: 'plomeria',
    displayName: 'Plomería',
    icon: 'water',
    description: 'Reparaciones de tuberías, grifos y sistemas de agua',
    color: '#007AFF',
  },
  techos: {
    id: 'techos',
    name: 'techos',
    displayName: 'Techos',
    icon: 'home',
    description: 'Reparación y mantenimiento de techos',
    color: '#FF9500',
  },
  carpinteria: {
    id: 'carpinteria',
    name: 'carpinteria',
    displayName: 'Carpintería',
    icon: 'hammer',
    description: 'Trabajos en madera y muebles',
    color: '#8E8E93',
  },
  electricidad: {
    id: 'electricidad',
    name: 'electricidad',
    displayName: 'Electricidad',
    icon: 'flash',
    description: 'Instalaciones y reparaciones eléctricas',
    color: '#FFD60A',
  },
  albanileria: {
    id: 'albanileria',
    name: 'albanileria',
    displayName: 'Albañilería',
    icon: 'square',
    description: 'Construcción y reparaciones en mampostería',
    color: '#A2845E',
  },
  pintura: {
    id: 'pintura',
    name: 'pintura',
    displayName: 'Pintura',
    icon: 'brush',
    description: 'Pintura de interiores y exteriores',
    color: '#FF2D55',
  },
  herreria: {
    id: 'herreria',
    name: 'herreria',
    displayName: 'Herrería',
    icon: 'construct',
    description: 'Trabajos en metal y soldadura',
    color: '#5856D6',
  },
  jardineria: {
    id: 'jardineria',
    name: 'jardineria',
    displayName: 'Jardinería',
    icon: 'leaf',
    description: 'Cuidado de plantas y jardines',
    color: '#34C759',
  },
  general: {
    id: 'general',
    name: 'general',
    displayName: 'General',
    icon: 'build',
    description: 'Reparaciones generales del hogar',
    color: '#8E8E93',
  },
};

export const CATEGORY_LIST = Object.values(CATEGORIES);




