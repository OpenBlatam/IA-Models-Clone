'use client';

import { memo, useMemo } from 'react';
import { Info } from 'lucide-react';
import { useAlerts } from '../hooks/useAlerts';
import { MAX_DISPLAYED_ALERTS } from '@/config/constants';
import VirtualizedList from '@/components/ui/VirtualizedList';
import AlertItem from './AlertItem';
import EmptyState from '@/components/ui/EmptyState';
import Card from '@/components/ui/Card';

const AlertsPanel = memo((): JSX.Element => {
  const { alerts } = useAlerts();

  const displayedAlerts = useMemo(
    () => alerts.slice(0, MAX_DISPLAYED_ALERTS),
    [alerts]
  );

  const renderAlert = useMemo(
    () => (alert: typeof alerts[0], index: number) => (
      <AlertItem key={`${alert.timestamp}-${index}`} alert={alert} />
    ),
    []
  );

  if (alerts.length === 0) {
    return (
      <Card title="Alerts">
        <EmptyState icon={Info} title="No alerts" />
      </Card>
    );
  }

  return (
    <Card title="Alerts">
      {displayedAlerts.length > 10 ? (
        <VirtualizedList
          items={displayedAlerts}
          renderItem={renderAlert}
          itemHeight={80}
          containerHeight={384}
        />
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {displayedAlerts.map((alert, index) => (
            <AlertItem key={`${alert.timestamp}-${index}`} alert={alert} />
          ))}
        </div>
      )}
    </Card>
  );
});

AlertsPanel.displayName = 'AlertsPanel';

export default AlertsPanel;
