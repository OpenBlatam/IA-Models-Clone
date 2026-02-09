import { cn } from '@/lib/utils';

type ButtonVariant = 'primary' | 'secondary' | 'danger';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  isLoading?: boolean;
}

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: 'btn-primary',
  secondary: 'btn-secondary',
  danger: 'btn-danger',
};

const LoadingSpinnerIcon = (): JSX.Element => (
  <svg
    className="animate-spin -ml-1 mr-3 h-5 w-5"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    />
  </svg>
);

const Button = ({
  children,
  variant = 'primary',
  isLoading = false,
  className,
  disabled,
  ...props
}: ButtonProps): JSX.Element => {
  const isDisabled = disabled || isLoading;
  const variantClass = VARIANT_CLASSES[variant];

  if (isLoading) {
    return (
      <button
        className={cn('btn', variantClass, className)}
        disabled={isDisabled}
        aria-busy="true"
        aria-label="Loading"
        {...props}
      >
        <span className="flex items-center">
          <LoadingSpinnerIcon />
          Loading...
        </span>
      </button>
    );
  }

  return (
    <button
      className={cn('btn', variantClass, className)}
      disabled={isDisabled}
      {...props}
    >
      {children}
    </button>
  );
};

export default Button;

