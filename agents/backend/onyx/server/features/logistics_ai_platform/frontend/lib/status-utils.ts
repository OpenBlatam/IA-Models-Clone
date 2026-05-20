import type { ShipmentStatus, ContainerStatus } from '@/types/api';

export const getStatusColor = (status: ShipmentStatus | ContainerStatus | string): string => {
  const statusMap: Record<string, string> = {
    delivered: 'bg-green-100 text-green-800',
    'in_transit': 'bg-blue-100 text-blue-800',
    booked: 'bg-green-100 text-green-800',
    pending: 'bg-yellow-100 text-yellow-800',
    quoted: 'bg-blue-100 text-blue-800',
    delayed: 'bg-orange-100 text-orange-800',
    cancelled: 'bg-red-100 text-red-800',
    exception: 'bg-red-100 text-red-800',
    loaded: 'bg-blue-100 text-blue-800',
    empty: 'bg-gray-100 text-gray-800',
    at_port: 'bg-purple-100 text-purple-800',
    at_terminal: 'bg-indigo-100 text-indigo-800',
    damaged: 'bg-red-100 text-red-800',
  };

  return statusMap[status] || 'bg-gray-100 text-gray-800';
};

export const getStatusBadgeVariant = (status: ShipmentStatus | ContainerStatus | string): 'success' | 'warning' | 'destructive' | 'info' | 'default' => {
  const statusMap: Record<string, 'success' | 'warning' | 'destructive' | 'info' | 'default'> = {
    delivered: 'success',
    booked: 'success',
    'in_transit': 'info',
    pending: 'warning',
    quoted: 'info',
    delayed: 'warning',
    cancelled: 'destructive',
    exception: 'destructive',
    loaded: 'info',
    empty: 'default',
    at_port: 'info',
    at_terminal: 'info',
    damaged: 'destructive',
  };

  return statusMap[status] || 'default';
};




