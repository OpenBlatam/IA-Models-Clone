import { PageContainer } from '@/components/layout/page-container';
import { PageHeader } from '@/components/layout/page-header';
import { AnalyticsDashboard } from '@/components/analytics-dashboard';

const AnalyticsPage = (): JSX.Element => {
  return (
    <PageContainer>
      <div className="max-w-6xl mx-auto">
        <PageHeader
          title="Analytics y Estadísticas"
          description="Análisis detallado del sistema y uso de manuales"
        />
        <AnalyticsDashboard />
      </div>
    </PageContainer>
  );
};

export default AnalyticsPage;

