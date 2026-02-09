/**
 * Enhanced validation table component
 */

'use client';

import React, { useState, useMemo } from 'react';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
  Button,
  Badge,
  Skeleton,
} from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { ValidationStatusBadge } from './ValidationStatusBadge';
import { ValidationPopover } from './ValidationPopover';
import { format } from 'date-fns';
import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import type { ValidationRead } from '@/lib/types';
import Link from 'next/link';

type SortField = 'created_at' | 'status' | 'id';
type SortDirection = 'asc' | 'desc';

export const ValidationTableEnhanced: React.FC = () => {
  const { data: validations, isLoading } = useValidations();
  const [sortField, setSortField] = useState<SortField>('created_at');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const sortedValidations = useMemo(() => {
    if (!validations) {
      return [];
    }

    return [...validations].sort((a, b) => {
      let aValue: any;
      let bValue: any;

      switch (sortField) {
        case 'created_at':
          aValue = new Date(a.created_at).getTime();
          bValue = new Date(b.created_at).getTime();
          break;
        case 'status':
          aValue = a.status;
          bValue = b.status;
          break;
        case 'id':
          aValue = a.id;
          bValue = b.id;
          break;
        default:
          return 0;
      }

      if (aValue < bValue) {
        return sortDirection === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortDirection === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [validations, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <ArrowUpDown className="h-4 w-4 ml-2" aria-hidden="true" />;
    }
    return sortDirection === 'asc' ? (
      <ArrowUp className="h-4 w-4 ml-2" aria-hidden="true" />
    ) : (
      <ArrowDown className="h-4 w-4 ml-2" aria-hidden="true" />
    );
  };

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} height={60} />
        ))}
      </div>
    );
  }

  if (!sortedValidations || sortedValidations.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No hay validaciones para mostrar
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleSort('id')}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleSort('id');
                }
              }}
              className="h-8 hover:bg-transparent"
              aria-label="Ordenar por ID"
              tabIndex={0}
            >
              ID
              <SortIcon field="id" />
            </Button>
          </TableHead>
          <TableHead>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleSort('status')}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleSort('status');
                }
              }}
              className="h-8 hover:bg-transparent"
              aria-label="Ordenar por estado"
              tabIndex={0}
            >
              Estado
              <SortIcon field="status" />
            </Button>
          </TableHead>
          <TableHead>Plataformas</TableHead>
          <TableHead>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleSort('created_at')}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleSort('created_at');
                }
              }}
              className="h-8 hover:bg-transparent"
              aria-label="Ordenar por fecha"
              tabIndex={0}
            >
              Fecha de Creación
              <SortIcon field="created_at" />
            </Button>
          </TableHead>
          <TableHead>Perfil</TableHead>
          <TableHead>Reporte</TableHead>
          <TableHead className="text-right">Acciones</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sortedValidations.map((validation) => (
          <TableRow key={validation.id}>
            <TableCell>
              <Link
                href={`/validations/${validation.id}`}
                className="font-mono text-sm hover:underline"
                aria-label={`Ver validación ${validation.id.slice(0, 8)}`}
                tabIndex={0}
              >
                {validation.id.slice(0, 8)}...
              </Link>
            </TableCell>
            <TableCell>
              <ValidationStatusBadge status={validation.status} />
            </TableCell>
            <TableCell>
              <div className="flex flex-wrap gap-1">
                {validation.connected_platforms.map((platform) => (
                  <Badge key={platform} variant="secondary" className="text-xs">
                    {platform}
                  </Badge>
                ))}
              </div>
            </TableCell>
            <TableCell>
              <time dateTime={validation.created_at}>
                {format(new Date(validation.created_at), 'dd/MM/yyyy HH:mm')}
              </time>
            </TableCell>
            <TableCell>
              <Badge variant={validation.has_profile ? 'success' : 'default'}>
                {validation.has_profile ? 'Sí' : 'No'}
              </Badge>
            </TableCell>
            <TableCell>
              <Badge variant={validation.has_report ? 'success' : 'default'}>
                {validation.has_report ? 'Sí' : 'No'}
              </Badge>
            </TableCell>
            <TableCell className="text-right">
              <ValidationPopover
                validation={validation}
                onView={() => {}}
                onDownload={() => {}}
                onShare={() => {}}
                onDelete={() => {}}
              />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};



