# Guía de Integración Stripe

## 💳 Configuración

### 1. Crear Cuenta Stripe

1. Ve a [Stripe Dashboard](https://dashboard.stripe.com/)
2. Crea una cuenta o inicia sesión
3. Obtén tus API keys (Test y Live)

### 2. Variables de Entorno

Agrega a tu `.env.local`:

```env
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_SECRET_KEY=sk_test_... (solo backend)
```

### 3. Instalación

```bash
npm install @stripe/stripe-js @stripe/react-stripe-js
```

## 🎯 Características Implementadas

### Página de Pricing (`/pricing`)
- Muestra 3 planes: Gratis, Pro, Enterprise
- Integración con Stripe Checkout
- Redirección automática al checkout
- Página de éxito después del pago

### Página de Suscripción (`/subscription`)
- Ver estado de suscripción actual
- Cancelar suscripción
- Gestionar facturación (Stripe Customer Portal)
- Ver fecha de renovación

### Componentes

#### PricingCard
```typescript
<PricingCard
  plan={plan}
  onSelect={handleSelectPlan}
  loading={loading}
  currentPlan={false}
/>
```

#### SubscriptionCard
```typescript
<SubscriptionCard
  subscription={subscription}
  onUpdate={fetchSubscription}
/>
```

## 🔌 API Endpoints Requeridos (Backend)

Tu backend debe implementar estos endpoints:

### 1. Crear Checkout Session
```
POST /api/stripe/create-checkout-session
Body: {
  priceId: string,
  successUrl: string,
  cancelUrl: string
}
Response: {
  sessionId: string
}
```

### 2. Crear Portal Session
```
POST /api/stripe/create-portal-session
Body: {
  customerId: string,
  returnUrl: string
}
Response: {
  url: string
}
```

### 3. Obtener Suscripción
```
GET /api/stripe/subscription
Response: {
  id: string,
  status: string,
  currentPeriodEnd: string,
  plan: Plan,
  cancelAtPeriodEnd: boolean
}
```

### 4. Cancelar Suscripción
```
POST /api/stripe/cancel-subscription
Body: {
  subscriptionId: string
}
Response: {
  success: boolean
}
```

### 5. Verificar Sesión
```
GET /api/stripe/verify-session?session_id=xxx
Response: {
  verified: boolean
}
```

## 📦 Estructura de Datos

### Plan
```typescript
interface Plan {
  id: string;
  name: string;
  description: string;
  price: number; // en centavos
  currency: string;
  interval: 'month' | 'year';
  features: string[];
  popular?: boolean;
  stripePriceId: string;
}
```

### Subscription
```typescript
interface Subscription {
  id: string;
  status: 'active' | 'canceled' | 'past_due' | 'unpaid';
  currentPeriodEnd: string;
  plan: Plan;
  cancelAtPeriodEnd: boolean;
}
```

## 🎨 Flujo de Pago

1. Usuario selecciona plan en `/pricing`
2. Se crea checkout session en backend
3. Redirección a Stripe Checkout
4. Usuario completa pago
5. Redirección a `/pricing/success`
6. Verificación de sesión
7. Activación de suscripción

## 🔒 Seguridad

- ✅ Nunca exponer secret key en frontend
- ✅ Validar sesiones en backend
- ✅ Verificar webhooks de Stripe
- ✅ Usar HTTPS en producción
- ✅ Validar estado de suscripción

## 🌐 Traducciones

Todas las páginas y componentes tienen traducciones en:
- Español
- Inglés
- Francés
- Portugués

## 🧪 Testing

### Modo Test
Usa las API keys de test de Stripe:
- `pk_test_...` para publishable key
- `sk_test_...` para secret key (backend)

### Tarjetas de Prueba
- Éxito: `4242 4242 4242 4242`
- Rechazada: `4000 0000 0000 0002`
- Requiere autenticación: `4000 0025 0000 3155`

## 📝 Notas

- Los precios están en centavos (2999 = $29.99)
- El plan gratuito no requiere Stripe
- Las suscripciones se renuevan automáticamente
- Los usuarios pueden cancelar en cualquier momento
- El acceso continúa hasta el final del período pagado

## 🐛 Troubleshooting

### Error: "Stripe publishable key not found"
- Verifica que `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` esté en `.env.local`
- Reinicia el servidor de desarrollo

### Error: "Checkout session creation failed"
- Verifica que el backend esté corriendo
- Verifica que el `priceId` sea válido en Stripe
- Revisa los logs del backend

### Redirección no funciona
- Verifica que las URLs de éxito/cancelación sean correctas
- Asegúrate de que el dominio esté permitido en Stripe Dashboard



