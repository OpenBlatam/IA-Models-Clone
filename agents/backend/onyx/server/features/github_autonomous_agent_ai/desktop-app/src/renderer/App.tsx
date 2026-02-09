import React, { useState, useEffect } from 'react';
import { Toaster } from './components/Toaster';
import { Layout } from './components/Layout';
import MainPage from './pages/MainPage';
import AgentControlPage from './pages/AgentControlPage';
import KanbanPage from './pages/KanbanPage';
import ContinuousAgentPage from './pages/ContinuousAgentPage';

type Page = 'main' | 'agent-control' | 'kanban' | 'continuous-agent';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>('main');
  const [appVersion, setAppVersion] = useState<string>('');

  useEffect(() => {
    // Get app version
    if (typeof window !== 'undefined' && window.electronAPI) {
      window.electronAPI.getVersion().then(setAppVersion);
    }
  }, []);

  const renderPage = () => {
    switch (currentPage) {
      case 'main':
        return <MainPage onNavigate={setCurrentPage} />;
      case 'agent-control':
        return <AgentControlPage onNavigate={setCurrentPage} />;
      case 'kanban':
        return <KanbanPage onNavigate={setCurrentPage} />;
      case 'continuous-agent':
        return <ContinuousAgentPage onNavigate={setCurrentPage} />;
      default:
        return <MainPage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
      {renderPage()}
      <Toaster />
    </Layout>
  );
};

export default App;
