/**
 * Table view for validations
 */

'use client';

import React from 'react';
import { DataTable, Badge } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { format } from 'date-fns';
import Link from 'next/link';
import type { ValidationRead } from '@/lib/types';
import { ExternalLink } from 'lucide-react';

export const ValidationTable: React.FC = () => {
  const { data: validations, isLoading } = useValidations();

  if (isLoading) {
    return <div className="text-center py-12">Cargando...</div>;
  }

  if (!validations || validations.length === 0) {
    return <div className="text-center py-12 text-muted-foreground">No hay validaciones</div>;
  }

  const getStatusVariant = (status: string): 'success' | 'destructive' | 'warning' | 'info' | 'default' => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'destructive';
      case 'running':
        return 'info';
      case 'cancelled':
        return 'warning';
      default:
        return 'default';
    }
  };

  const columns = [
    {
      id: 'id',
      header: 'ID',
      accessor: (row: ValidationRead) => (
        <Link
          href={`/validations/${row.id}`}
          className="flex items-center gap-2 text-primary hover:underline"
          aria-label={`Ver validación ${row.id.slice(0, 8)}`}
          tabIndex={0}
        >
          #{row.id.slice(0, 8)}
          <ExternalLink className="h-3 w-3" aria-hidden="true" />
        </Link>
      ),
      sortable: true,
    },
    {
      id: 'status',
      header: 'Estado',
      accessor: (row: ValidationRead) => (
        <Badge variant={getStatusVariant(row.status)}>{row.status}</Badge>
      ),
      sortable: true,
    },
    {
      id: 'platforms',
      header: 'Plataformas',
      accessor: (row: ValidationRead) => (
        <span className="text-sm">{row.connected_platforms.length}</span>
      ),
      sortable: true,
    },
    {
      id: 'created_at',
      header: 'Creada',
      accessor: (row: ValidationRead) => (
        <time dateTime={row.created_at}>
          {format(new Date(row.created_at), 'PPp')}
        </time>
      ),
      sortable: true,
    },
    {
      id: 'completed_at',
      header: 'Completada',
      accessor: (row: ValidationRead) =>
        row.completed_at ? (
          <time dateTime={row.completed_at}>
            {format(new Date(row.completed_at), 'PPp')}
          </time>
        ) : (
          <span className="text-muted-foreground">-</span>
        ),
      sortable: true,
    },
    {
      id: 'has_profile',
      header: 'Perfil',
      accessor: (row: ValidationRead) => (
        <Badge variant={row.has_profile ? 'success' : 'default'}>
          {row.has_profile ? 'Sí' : 'No'}
        </Badge>
      ),
      sortable: true,
    },
    {
      id: 'has_report',
      header: 'Reporte',
      accessor: (row: ValidationRead) => (
        <Badge variant={row.has_report ? 'success' : 'default'}>
          {row.has_report ? 'Sí' : 'No'}
        </Badge>
      ),
      sortable: true,
    },
  ];

  return <DataTable data={validations} columns={columns} pageSize={10} />;
};




