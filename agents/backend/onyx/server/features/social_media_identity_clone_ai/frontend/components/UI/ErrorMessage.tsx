interface ErrorMessageProps {
  message: string;
  className?: string;
}

const ErrorMessage = ({ message, className = '' }: ErrorMessageProps): JSX.Element => {
  if (!message) {
    return <></>;
  }

  return (
    <div
      className={`text-sm text-red-600 ${className}`}
      role="alert"
      aria-live="polite"
    >
      {message}
    </div>
  );
};

export default ErrorMessage;



