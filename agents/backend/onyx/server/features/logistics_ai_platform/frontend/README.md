# Logistics AI Platform - Frontend

Next.js 15.3 frontend application for the Logistics AI Platform with TypeScript, Turbopack, authentication, i18n, and Stripe integration.

## Features

- ✅ Next.js 15.3 with Turbopack
- ✅ TypeScript
- ✅ Google and Email authentication (NextAuth.js)
- ✅ Internationalization (i18n) with English and Spanish
- ✅ Stripe payment integration
- ✅ Complete API integration for all backend features:
  - Quotes management
  - Bookings management
  - Shipments tracking and management
  - Container management
  - Invoice management with Stripe payments
  - Insurance management
  - Document upload and management
  - Alerts and notifications
  - Reports and dashboard
  - Public tracking

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create a `.env.local` file:
```env
# NextAuth
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# API
NEXT_PUBLIC_API_URL=http://localhost:8030

# Stripe
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=your-stripe-publishable-key
STRIPE_SECRET_KEY=your-stripe-secret-key
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── [locale]/          # Internationalized routes
│   │   ├── auth/          # Authentication pages
│   │   ├── dashboard/     # Dashboard page
│   │   ├── quotes/        # Quotes pages
│   │   ├── bookings/      # Bookings pages
│   │   ├── shipments/     # Shipments pages
│   │   ├── tracking/      # Tracking page
│   │   ├── invoices/      # Invoices pages
│   │   └── alerts/        # Alerts page
│   ├── api/               # API routes
│   │   ├── auth/          # NextAuth routes
│   │   └── stripe/        # Stripe integration
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── ui/               # UI components (Button, Card, Input, etc.)
│   └── layout/           # Layout components (Navbar, etc.)
├── lib/                  # Utility libraries
│   ├── api/              # API client functions
│   ├── auth.ts           # NextAuth configuration
│   ├── stripe.ts         # Stripe configuration
│   └── utils.ts         # Utility functions
├── types/                # TypeScript type definitions
├── messages/             # i18n translation files
│   ├── en.json          # English translations
│   └── es.json          # Spanish translations
└── i18n/                # i18n configuration
```

## Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Authentication

The application supports two authentication methods:

1. **Email/Password** - Traditional email and password authentication
2. **Google OAuth** - Sign in with Google account

Both methods are configured through NextAuth.js and require proper environment variables.

## Internationalization

The application supports two languages:
- English (en) - Default
- Spanish (es)

Language switching can be implemented in the navbar component.

## Stripe Integration

Stripe is integrated for invoice payments. To use Stripe:

1. Set up your Stripe account
2. Add your Stripe keys to `.env.local`
3. The payment flow is handled in the invoices page

## API Integration

All backend API endpoints are integrated through the API client layer in `lib/api/`. The API client automatically handles:
- Authentication tokens
- Error handling
- Request/response formatting

## Technologies Used

- **Next.js 15.3** - React framework with App Router
- **TypeScript** - Type safety
- **Turbopack** - Fast bundler
- **NextAuth.js** - Authentication
- **next-intl** - Internationalization
- **Stripe** - Payment processing
- **TanStack Query** - Data fetching and caching
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

## License

MIT
