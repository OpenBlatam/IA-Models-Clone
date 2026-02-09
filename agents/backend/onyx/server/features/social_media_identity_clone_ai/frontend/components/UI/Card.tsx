import { cn } from '@/lib/utils';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  role?: string;
}

const Card = ({ children, className, title, role }: CardProps): JSX.Element => {
  if (!title) {
    return (
      <div className={cn('card', className)} role={role}>
        {children}
      </div>
    );
  }

  return (
    <div className={cn('card', className)} role={role}>
      <h3 className="text-xl font-semibold mb-4">{title}</h3>
      {children}
    </div>
  );
};

export default Card;

