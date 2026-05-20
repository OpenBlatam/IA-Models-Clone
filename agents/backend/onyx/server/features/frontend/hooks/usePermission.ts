'use client';

import { useMemo } from 'react';
import { permissionManager } from '@/lib/permissions';

export function usePermission(permission: string): boolean {
  return useMemo(() => {
    return permissionManager.hasPermission(permission);
  }, [permission]);
}

export function usePermissions(permissions: string[]): {
  hasAll: boolean;
  hasAny: boolean;
  hasPermission: (permission: string) => boolean;
} {
  return useMemo(() => {
    return {
      hasAll: permissionManager.hasAllPermissions(permissions),
      hasAny: permissionManager.hasAnyPermission(permissions),
      hasPermission: (permission: string) => permissionManager.hasPermission(permission),
    };
  }, [permissions]);
}

export function useRole(): string | null {
  return useMemo(() => {
    return permissionManager.getRole();
  }, []);
}

