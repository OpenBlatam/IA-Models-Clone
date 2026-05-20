type Permission = string;
type Role = string;

interface PermissionConfig {
  roles: Record<Role, Permission[]>;
  defaultRole?: Role;
}

class PermissionManager {
  private config: PermissionConfig;
  private currentRole: Role | null = null;

  constructor(config: PermissionConfig) {
    this.config = config;
    this.currentRole = config.defaultRole || null;
  }

  setRole(role: Role): void {
    if (role in this.config.roles) {
      this.currentRole = role;
    }
  }

  getRole(): Role | null {
    return this.currentRole;
  }

  hasPermission(permission: Permission): boolean {
    if (!this.currentRole) return false;
    const rolePermissions = this.config.roles[this.currentRole] || [];
    return rolePermissions.includes(permission);
  }

  hasAnyPermission(permissions: Permission[]): boolean {
    return permissions.some((permission) => this.hasPermission(permission));
  }

  hasAllPermissions(permissions: Permission[]): boolean {
    return permissions.every((permission) => this.hasPermission(permission));
  }

  getPermissions(): Permission[] {
    if (!this.currentRole) return [];
    return this.config.roles[this.currentRole] || [];
  }

  clearRole(): void {
    this.currentRole = null;
  }
}

// Default permissions configuration
const defaultConfig: PermissionConfig = {
  roles: {
    admin: ['*'], // All permissions
    user: ['read', 'write', 'delete'],
    viewer: ['read'],
  },
  defaultRole: 'viewer',
};

export const permissionManager = new PermissionManager(defaultConfig);

