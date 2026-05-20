import React from 'react';
import { Button } from '../components/ui/Button';
import { useTasks, useAgents } from '../hooks/useAPI';

type Page = 'main' | 'agent-control' | 'kanban' | 'continuous-agent';

interface MainPageProps {
  onNavigate: (page: Page) => void;
}

const MainPage: React.FC<MainPageProps> = ({ onNavigate }) => {
  const { tasks, loading: tasksLoading } = useTasks();
  const { agents, loading: agentsLoading } = useAgents();

  const stats = {
    tasks: {
      total: tasks.length,
      pending: tasks.filter((t) => t.status === 'pending').length,
      inProgress: tasks.filter((t) => t.status === 'in_progress').length,
      completed: tasks.filter((t) => t.status === 'completed').length,
    },
    agents: {
      total: agents.length,
      active: agents.filter((a) => a.status === 'active').length,
    },
  };

  return (
    <div className="min-h-screen bg-white text-black">
      <div className="max-w-7xl mx-auto px-5 md:px-6 lg:px-8 py-10 md:py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-normal text-black mb-6">
            GitHub Autonomous Agent AI
          </h1>
          <p className="text-xl text-gray-500 mb-8 max-w-2xl mx-auto">
            Agente autónomo inteligente para automatizar tareas en GitHub
          </p>
          <div className="flex gap-4 justify-center">
            <Button
              variant="primary"
              size="lg"
              onClick={() => onNavigate('continuous-agent')}
            >
              Comenzar
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => onNavigate('agent-control')}
            >
              Agent Control
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow">
            <div className="text-3xl font-normal text-black">{stats.tasks.total}</div>
            <div className="text-sm text-gray-500 mt-1">Total de Tareas</div>
          </div>
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg hover:shadow-sm transition-shadow">
            <div className="flex items-center justify-between mb-1">
              <div className="text-xs text-yellow-700 font-medium">Pendientes</div>
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
            </div>
            <div className="text-xl font-normal text-yellow-800">{stats.tasks.pending}</div>
          </div>
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg hover:shadow-sm transition-shadow">
            <div className="flex items-center justify-between mb-1">
              <div className="text-xs text-blue-700 font-medium">En Progreso</div>
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            </div>
            <div className="text-xl font-normal text-blue-800">{stats.tasks.inProgress}</div>
          </div>
          <div className="p-3 bg-green-50 border border-green-200 rounded-lg hover:shadow-sm transition-shadow">
            <div className="flex items-center justify-between mb-1">
              <div className="text-xs text-green-700 font-medium">Agentes Activos</div>
              <svg className="w-3 h-3 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="text-xl font-normal text-green-800">{stats.agents.active}</div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div 
            className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow cursor-pointer"
            onClick={() => onNavigate('continuous-agent')}
          >
            <div className="text-2xl mb-2">🤖</div>
            <h3 className="font-normal text-black mb-1">Agente Continuo</h3>
            <p className="text-sm text-gray-500">
              Gestiona y monitorea tus agentes autónomos
            </p>
          </div>
          <div 
            className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow cursor-pointer"
            onClick={() => onNavigate('kanban')}
          >
            <div className="text-2xl mb-2">📋</div>
            <h3 className="font-normal text-black mb-1">Kanban Board</h3>
            <p className="text-sm text-gray-500">
              Visualiza tus tareas en un tablero Kanban
            </p>
          </div>
          <div 
            className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow cursor-pointer"
            onClick={() => onNavigate('agent-control')}
          >
            <div className="text-2xl mb-2">⚙️</div>
            <h3 className="font-normal text-black mb-1">Agent Control</h3>
            <p className="text-sm text-gray-500">
              Configura y controla el agente
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainPage;

