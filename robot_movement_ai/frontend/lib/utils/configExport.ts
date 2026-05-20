export interface AppConfig {
  theme: 'dark' | 'light';
  pollingInterval: number;
  autoReconnect: boolean;
  apiUrl: string;
  preferences: {
    [key: string]: any;
  };
}

export function exportConfig(config: AppConfig): void {
  const dataStr = JSON.stringify(config, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `robot-config-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

export async function importConfig(): Promise<AppConfig | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const config = JSON.parse(event.target?.result as string) as AppConfig;
            resolve(config);
          } catch (error) {
            console.error('Error parsing config file:', error);
            resolve(null);
          }
        };
        reader.readAsText(file);
      } else {
        resolve(null);
      }
    };
    input.click();
  });
}

export function getDefaultConfig(): AppConfig {
  return {
    theme: 'dark',
    pollingInterval: 2000,
    autoReconnect: true,
    apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8010',
    preferences: {},
  };
}

