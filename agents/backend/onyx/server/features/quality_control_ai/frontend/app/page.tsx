'use client';

import { useEffect, Suspense, memo, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { useCamera } from '@/modules/camera/hooks/useCamera';
import { useAlerts } from '@/modules/alerts/hooks/useAlerts';
import { useInspection } from '@/modules/inspection/hooks/useInspection';
import { useQualityControlStore } from '@/lib/store';
import { useToast } from '@/lib/hooks/useToast';
import Layout from '@/components/layout/Layout';
import LoadingSpinner from '@/components/LoadingSpinner';
import { INSPECTION_INTERVAL } from '@/config/constants';

const CameraView = dynamic(() => import('@/modules/camera/components/CameraView'), {
  loading: () => <LoadingSpinner />,
  ssr: false,
});

const InspectionResults = dynamic(
  () => import('@/modules/inspection/components/InspectionResults'),
  {
    loading: () => <LoadingSpinner />,
  }
);

const ImageUpload = dynamic(() => import('@/modules/inspection/components/ImageUpload'), {
  loading: () => <LoadingSpinner />,
  ssr: false,
});

const ControlPanel = dynamic(() => import('@/modules/control/components/ControlPanel'), {
  loading: () => <LoadingSpinner />,
});

const AlertsPanel = dynamic(() => import('@/modules/alerts/components/AlertsPanel'), {
  loading: () => <LoadingSpinner />,
});

const StatisticsPanel = dynamic(
  () => import('@/modules/statistics/components/StatisticsPanel'),
  {
    loading: () => <LoadingSpinner />,
  }
);

const Home = memo((): JSX.Element => {
  const { setCameraInfo, setAlerts, isInspecting, setCurrentResult, addToHistory } =
    useQualityControlStore();
  const { cameraInfo } = useCamera();
  const { alerts } = useAlerts();
  const { inspectFrame } = useInspection();
  const toast = useToast();

  useEffect(() => {
    if (cameraInfo) {
      setCameraInfo(cameraInfo);
    }
  }, [cameraInfo, setCameraInfo]);

  useEffect(() => {
    if (alerts && alerts.length > 0) {
      setAlerts(alerts);
    }
  }, [alerts, setAlerts]);

  useEffect(() => {
    if (!isInspecting) return;

    const interval = setInterval(async () => {
      try {
        const result = await inspectFrame();
        if (result) {
          setCurrentResult(result);
          addToHistory(result);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to inspect frame';
        toast.error(message);
      }
    }, INSPECTION_INTERVAL);

    return () => clearInterval(interval);
  }, [isInspecting, inspectFrame, setCurrentResult, addToHistory, toast]);

  const mainContent = useMemo(
    () => (
      <div className="lg:col-span-2 space-y-6">
        <Suspense fallback={<LoadingSpinner />}>
          <CameraView />
        </Suspense>
        <Suspense fallback={<LoadingSpinner />}>
          <ImageUpload />
        </Suspense>
        <Suspense fallback={<LoadingSpinner />}>
          <InspectionResults />
        </Suspense>
      </div>
    ),
    []
  );

  const sidebarContent = useMemo(
    () => (
      <div className="space-y-6">
        <Suspense fallback={<LoadingSpinner />}>
          <ControlPanel />
        </Suspense>
        <Suspense fallback={<LoadingSpinner />}>
          <AlertsPanel />
        </Suspense>
        <Suspense fallback={<LoadingSpinner />}>
          <StatisticsPanel />
        </Suspense>
      </div>
    ),
    []
  );

  return (
    <Layout>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {mainContent}
        {sidebarContent}
      </div>
    </Layout>
  );
});

Home.displayName = 'Home';

export default Home;
