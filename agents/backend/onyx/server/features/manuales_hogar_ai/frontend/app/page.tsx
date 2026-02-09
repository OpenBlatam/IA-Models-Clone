import { PageContainer } from '@/components/layout/page-container';
import { PageHeader } from '@/components/layout/page-header';
import { ManualGenerator } from '@/components/manual-generator';
import { RecentManuals } from '@/components/recent-manuals';

const HomePage = (): JSX.Element => {
  return (
    <PageContainer>
      <div className="max-w-4xl mx-auto">
        <PageHeader
          title="Manuales Hogar AI"
          description="Genera manuales paso a paso tipo LEGO para resolver problemas del hogar"
        />
        <ManualGenerator />
        <RecentManuals />
      </div>
    </PageContainer>
  );
};

export default HomePage;

