'use client';

import { Button } from '@/components/ui/button';
import { Trash2, Eye, Edit } from 'lucide-react';
import Link from 'next/link';

interface ActionButtonsProps {
  viewHref?: string;
  editHref?: string;
  onDelete?: () => void;
  onEdit?: () => void;
  onView?: () => void;
  showView?: boolean;
  showEdit?: boolean;
  showDelete?: boolean;
  deleteLabel?: string;
}

const ActionButtons = ({
  viewHref,
  editHref,
  onDelete,
  onEdit,
  onView,
  showView = true,
  showEdit = false,
  showDelete = true,
  deleteLabel = 'Eliminar',
}: ActionButtonsProps) => {
  return (
    <div className="flex gap-2">
      {showView && (
        <>
          {viewHref ? (
            <Link href={viewHref} className="flex-1">
              <Button variant="secondary" size="sm" className="w-full">
                <Eye className="w-4 h-4 mr-2" />
                Ver Detalles
              </Button>
            </Link>
          ) : (
            onView && (
              <Button variant="secondary" size="sm" onClick={onView} className="flex-1">
                <Eye className="w-4 h-4 mr-2" />
                Ver Detalles
              </Button>
            )
          )}
        </>
      )}
      {showEdit && (
        <>
          {editHref ? (
            <Link href={editHref} className="flex-1">
              <Button variant="secondary" size="sm" className="w-full">
                <Edit className="w-4 h-4 mr-2" />
                Editar
              </Button>
            </Link>
          ) : (
            onEdit && (
              <Button variant="secondary" size="sm" onClick={onEdit} className="flex-1">
                <Edit className="w-4 h-4 mr-2" />
                Editar
              </Button>
            )
          )}
        </>
      )}
      {showDelete && onDelete && (
        <Button variant="danger" size="sm" onClick={onDelete} aria-label={deleteLabel}>
          <Trash2 className="w-4 h-4" />
        </Button>
      )}
    </div>
  );
};

export { ActionButtons };

