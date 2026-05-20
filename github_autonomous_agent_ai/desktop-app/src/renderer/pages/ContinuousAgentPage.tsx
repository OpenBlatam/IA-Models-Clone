import React, { useState } from 'react';
import { useContinuousAgents } from '../hooks/useContinuousAgents';
import { Button } from '../components/ui/Button';
import { GithubAuth } from '../components/GithubAuth';
import { AgentCard } from '../components/AgentCard';
import { CreateAgentModal } from '../components/CreateAgentModal';
import { GitHubUser } from '../lib/github-api';
import type { ContinuousAgent, CreateAgentRequest } from '../types/agent';

type Page = 'main' | 'agent-control' | 'kanban' | 'continuous-agent';

interface ContinuousAgentPageProps {
  onNavigate: (page: Page) => void;
}

const ContinuousAgentPage: React.FC<ContinuousAgentPageProps> = ({ onNavigate }) => {
  const {
    agents,
    loading,
    error,
    createAgent,
    updateAgent,
    deleteAgent,
    toggleAgent,
  } = useContinuousAgents({ autoRefresh: true, refreshInterval: 5000 });
  
  const [githubUser, setGithubUser] = useState<GitHubUser | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingAgent, setEditingAgent] = useState<ContinuousAgent | null>(null);

  const handleAuthSuccess = (user: GitHubUser) => {
    setGithubUser(user);
  };

  const handleAuthError = (error: string) => {
    console.error('Auth error:', error);
  };

  const handleCreateAgent = async (request: CreateAgentRequest) => {
    await createAgent(request);
    setShowCreateModal(false);
  };

  const handleEditAgent = (agent: ContinuousAgent) => {
    setEditingAgent(agent);
    // TODO: Open edit modal
  };

  const handleDeleteAgent = async (agentId: string) => {
    if (window.confirm('¿Estás seguro de eliminar este agente?')) {
      await deleteAgent(agentId);
    }
  };

  const handleToggleAgent = async (agentId: string) => {
    await toggleAgent(agentId);
  };

  return (
    <div className="min-h-screen bg-white text-black">
      <div className="max-w-7xl mx-auto px-5 md:px-6 lg:px-8 py-10 md:py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl md:text-4xl font-normal text-black">Agentes Continuos</h1>
            <p className="text-sm text-gray-500 mt-2">
              Gestiona y monitorea tus agentes autónomos de GitHub
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* GitHub Auth */}
          <div className="lg:col-span-1">
            <GithubAuth
              onAuthSuccess={handleAuthSuccess}
              onAuthError={handleAuthError}
            />
          </div>

          {/* Agents List */}
          <div className="lg:col-span-2">
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="text-black text-xl font-normal mb-1">Agentes Activos</h3>
                  <p className="text-sm text-gray-500">{`${agents.length} agente(s) configurado(s)`}</p>
                </div>
                <div className="ml-4">
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => setShowCreateModal(true)}
                  >
                    + Crear Agente
                  </Button>
                </div>
              </div>
              <div>
                {loading ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-black"></div>
                  </div>
                ) : error ? (
                  <div className="text-center py-8 text-red-600">
                    <p>{error}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={fetchAgents}
                      className="mt-4"
                    >
                      Reintentar
                    </Button>
                  </div>
                ) : agents.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <div className="text-4xl mb-4">🤖</div>
                    <p className="text-lg font-medium mb-2">No hay agentes configurados</p>
                    <p className="text-sm mb-4">
                      Crea tu primer agente para comenzar a automatizar tareas
                    </p>
                    <Button
                      variant="primary"
                      onClick={() => setShowCreateModal(true)}
                    >
                      Crear Agente
                    </Button>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {agents.map((agent) => (
                      <AgentCard
                        key={agent.id}
                        agent={agent}
                        onToggle={handleToggleAgent}
                        onDelete={handleDeleteAgent}
                        onEdit={handleEditAgent}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Create Agent Modal */}
      <CreateAgentModal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreate={handleCreateAgent}
      />
    </div>
  );
};

export default ContinuousAgentPage;
