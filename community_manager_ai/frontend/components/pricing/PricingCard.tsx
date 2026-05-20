'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Check } from 'lucide-react';
import { Plan } from '@/types/stripe';
import { cn } from '@/lib/utils';

interface PricingCardProps {
  plan: Plan;
  onSelect: (plan: Plan) => void;
  loading?: boolean;
  currentPlan?: boolean;
}

export const PricingCard = ({ plan, onSelect, loading = false, currentPlan = false }: PricingCardProps) => {
  const handleSelect = () => {
    onSelect(plan);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleSelect();
    }
  };

  const formatPrice = (price: number, currency: string) => {
    return new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: currency.toUpperCase(),
    }).format(price / 100);
  };

  return (
    <Card
      className={cn(
        'relative flex flex-col',
        plan.popular && 'border-primary-500 shadow-lg scale-105',
        currentPlan && 'border-green-500'
      )}
    >
      {plan.popular && (
        <div className="absolute -top-4 left-1/2 -translate-x-1/2">
          <Badge variant="success" size="lg">
            Popular
          </Badge>
        </div>
      )}
      {currentPlan && (
        <div className="absolute -top-4 right-4">
          <Badge variant="success" size="sm">
            Actual
          </Badge>
        </div>
      )}
      <CardHeader className="text-center">
        <CardTitle className="text-2xl">{plan.name}</CardTitle>
        <div className="mt-4">
          <span className="text-4xl font-bold text-gray-900 dark:text-gray-100">
            {formatPrice(plan.price, plan.currency)}
          </span>
          <span className="text-gray-600 dark:text-gray-400">
            /{plan.interval === 'month' ? 'mes' : 'año'}
          </span>
        </div>
        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">{plan.description}</p>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col">
        <ul className="mb-6 space-y-3 flex-1">
          {plan.features.map((feature, index) => (
            <li key={index} className="flex items-start gap-2">
              <Check className="h-5 w-5 flex-shrink-0 text-green-600 dark:text-green-400 mt-0.5" />
              <span className="text-sm text-gray-700 dark:text-gray-300">{feature}</span>
            </li>
          ))}
        </ul>
        <Button
          variant={plan.popular ? 'primary' : 'secondary'}
          size="lg"
          className="w-full"
          onClick={handleSelect}
          onKeyDown={handleKeyDown}
          disabled={loading || currentPlan}
          aria-label={`Seleccionar plan ${plan.name}`}
          tabIndex={0}
        >
          {currentPlan ? 'Plan Actual' : loading ? 'Procesando...' : 'Seleccionar Plan'}
        </Button>
      </CardContent>
    </Card>
  );
};



