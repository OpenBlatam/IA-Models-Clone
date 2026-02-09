import Button from './Button';

interface EmptyStateProps {
  title: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  actionHref?: string;
}

const EmptyState = ({
  title,
  description,
  actionLabel,
  onAction,
  actionHref,
}: EmptyStateProps): JSX.Element => {
  return (
    <div className="text-center py-12" role="status" aria-live="polite">
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
      {description && <p className="text-gray-600 mb-6">{description}</p>}
      {(actionLabel && (onAction || actionHref)) && (
        <div>
          {onAction ? (
            <Button onClick={onAction} aria-label={actionLabel}>
              {actionLabel}
            </Button>
          ) : (
            <a
              href={actionHref}
              className="btn btn-primary"
              tabIndex={0}
              aria-label={actionLabel}
            >
              {actionLabel}
            </a>
          )}
        </div>
      )}
    </div>
  );
};

export default EmptyState;



