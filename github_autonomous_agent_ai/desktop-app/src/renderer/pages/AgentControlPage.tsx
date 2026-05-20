import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { ModelSelector } from '../components/ModelSelector';
import { setAPIKey, removeAPIKey, hasAPIKey } from '../lib/api-client';
import { toast } from 'sonner';
import { APP_CONFIG } from '../../shared/config';
import type { AIModel } from '../lib/ai-providers';

type Page = 'main' | 'agent-control' | 'kanban' | 'continuous-agent';

interface AgentControlPageProps {
  onNavigate: (page: Page) => void;
}

const AgentControlPage: React.FC<AgentControlPageProps> = ({ onNavigate }) => {
  const [apiKey, setApiKey] = useState('');
  const [apiUrl, setApiUrl] = useState(APP_CONFIG.apiBaseUrl);
  const [hasKey, setHasKey] = useState(hasAPIKey());
  const [selectedModel, setSelectedModel] = useState<AIModel>('deepseek-chat');

  // Guardar modelo seleccionado en localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('selected_ai_model', selectedModel);
    }
  }, [selectedModel]);

  // Cargar modelo guardado al iniciar
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedModel = localStorage.getItem('selected_ai_model') as AIModel | null;
      if (savedModel) {
        setSelectedModel(savedModel);
      }
    }
  }, []);

  const handleSaveAPIKey = () => {
    if (!apiKey.trim()) {
      toast.error('Por favor, ingresa un API key válido');
      return;
    }
    setAPIKey(apiKey.trim());
    setHasKey(true);
    setApiKey('');
    toast.success('API key guardada exitosamente');
  };

  const handleRemoveAPIKey = () => {
    removeAPIKey();
    setHasKey(false);
    toast.success('API key eliminada');
  };

  return (
    <div className="min-h-screen bg-white text-black">
      <div className="max-w-7xl mx-auto px-5 md:px-6 lg:px-8 py-10 md:py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl md:text-4xl font-normal text-black">Agent Control</h1>
            <p className="text-sm text-gray-500 mt-2">
              Configura y gestiona el agente autónomo de GitHub
            </p>
          </div>
        </div>

        <div className="space-y-6">
          {/* API Configuration */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="mb-4">
              <h3 className="text-black text-xl font-normal mb-1">Configuración de API</h3>
              <p className="text-sm text-gray-500">Configura la conexión con el backend</p>
            </div>
            <div className="space-y-4">
              <Input
                label="URL del Backend"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="http://localhost:8030"
                helperText="URL base del servidor backend"
                fullWidth
              />
              {hasKey ? (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-green-900">API Key configurada</p>
                      <p className="text-sm text-green-700 mt-1">
                        La API key está guardada de forma segura
                      </p>
                    </div>
                    <Button variant="danger" size="sm" onClick={handleRemoveAPIKey}>
                      Eliminar
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <Input
                    type="password"
                    label="API Key"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Ingresa tu API key"
                    helperText="Token de autenticación para el backend"
                    fullWidth
                  />
                  <Button variant="primary" onClick={handleSaveAPIKey} fullWidth>
                    Guardar API Key
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* AI Model Selection */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="mb-4">
              <h3 className="text-black text-xl font-normal mb-1">Configuración de IA</h3>
              <p className="text-sm text-gray-500">Selecciona el modelo de IA a utilizar</p>
            </div>
            <div>
              <ModelSelector
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
              />
              <p className="text-xs text-gray-500 mt-2">
                El modelo seleccionado se usará para las operaciones del agente
              </p>
            </div>
          </div>

          {/* Connection Status */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="mb-4">
              <h3 className="text-black text-xl font-normal mb-1">Estado de Conexión</h3>
              <p className="text-sm text-gray-500">Verifica la conexión con el backend</p>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-black">Backend</p>
                <p className="text-sm text-gray-500">{apiUrl}</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="text-sm text-black">Conectado</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentControlPage;

