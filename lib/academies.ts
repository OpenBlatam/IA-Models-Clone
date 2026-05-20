import { Academy } from "@/lib/types/academy";
import { academies } from "./data/academies";

export { academies };

export function getAcademyById(id: string): Academy | undefined {
  return academies.find(academy => academy.id === id);
}

export function getClassById(academyId: string, classId: string) {
  const academy = getAcademyById(academyId);
  return academy?.classes.find(c => c.id === classId);
}

export function getAcademiesByCategory(category: string): Academy[] {
  return academies.filter((academy) => academy.category === category);
}

export function getAcademiesByLevel(level: string): Academy[] {
  return academies.filter((academy) => academy.level === level);
} 