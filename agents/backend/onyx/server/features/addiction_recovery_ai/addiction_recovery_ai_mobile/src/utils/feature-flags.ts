interface FeatureFlags {
  [key: string]: boolean;
}

class FeatureFlagManager {
  private flags: FeatureFlags = {};

  setFlags(flags: FeatureFlags): void {
    this.flags = { ...this.flags, ...flags };
  }

  isEnabled(flag: string): boolean {
    return this.flags[flag] ?? false;
  }

  enable(flag: string): void {
    this.flags[flag] = true;
  }

  disable(flag: string): void {
    this.flags[flag] = false;
  }

  getAllFlags(): FeatureFlags {
    return { ...this.flags };
  }
}

export const featureFlags = new FeatureFlagManager();

// Initialize feature flags from environment or config
if (typeof process !== 'undefined' && process.env) {
  const envFlags: FeatureFlags = {};
  
  Object.keys(process.env).forEach((key) => {
    if (key.startsWith('FEATURE_')) {
      const flagName = key.replace('FEATURE_', '').toLowerCase();
      envFlags[flagName] = process.env[key] === 'true';
    }
  });

  featureFlags.setFlags(envFlags);
}

