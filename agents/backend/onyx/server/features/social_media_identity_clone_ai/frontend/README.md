# Social Media Identity Clone AI - Frontend

Frontend application built with Next.js, React, TypeScript, and TailwindCSS that provides a complete interface for the Social Media Identity Clone AI backend API.

## Features

- ✅ **Profile Extraction** - Extract profiles from TikTok, Instagram, and YouTube
- ✅ **Identity Building** - Build complete identity profiles from social media data
- ✅ **Content Generation** - Generate authentic content based on cloned identities
- ✅ **Dashboard** - View analytics, metrics, and system overview
- ✅ **Identity Management** - Browse and manage all cloned identities
- ✅ **Templates** - Create and manage content templates
- ✅ **Tasks** - Monitor async tasks and their status
- ✅ **Alerts** - View and manage system alerts

## Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **TailwindCSS** - Utility-first CSS framework
- **React Query** - Data fetching and caching
- **Axios** - HTTP client
- **React Hook Form** - Form management

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running (default: http://localhost:8000)

### Installation

```bash
# Install dependencies
npm install

# Set environment variables
cp .env.example .env.local
# Edit .env.local with your API URL

# Run development server
npm run dev
```

The application will be available at http://localhost:3000

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── page.tsx           # Home page
│   ├── extract-profile/   # Profile extraction page
│   ├── build-identity/    # Identity building page
│   ├── generate-content/  # Content generation page
│   ├── dashboard/         # Dashboard page
│   ├── identities/        # Identity management pages
│   ├── templates/         # Template management
│   ├── tasks/             # Task monitoring
│   └── alerts/            # Alert management
├── components/            # React components
│   ├── Layout/           # Layout components
│   └── UI/               # Reusable UI components
├── lib/                   # Utilities and services
│   ├── api/              # API client
│   └── utils.ts          # Utility functions
├── types/                 # TypeScript type definitions
└── public/                # Static assets
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Features Overview

### Profile Extraction
Extract complete profiles from social media platforms including:
- Profile metadata (bio, followers, etc.)
- Videos and posts
- Comments and engagement data

### Identity Building
Combine multiple social media profiles to create a comprehensive identity profile with:
- Content analysis
- Personality traits
- Communication style
- Topics and themes

### Content Generation
Generate authentic content based on cloned identities:
- Platform-specific content (Instagram posts, TikTok scripts, YouTube descriptions)
- Style and tone matching
- Hashtag suggestions
- Content validation

### Dashboard
Monitor system health and metrics:
- Total identities and content
- Content by platform
- System metrics and counters
- Recent activity

## API Integration

The frontend uses a centralized API client (`lib/api/client.ts`) that provides methods for all backend endpoints. The client handles:
- Request/response transformation
- Error handling
- API key management
- Type safety

## Styling

The application uses TailwindCSS with custom utility classes defined in `app/globals.css`. Components follow a consistent design system with:
- Primary color scheme
- Responsive design
- Accessible components
- Modern UI/UX patterns

## Accessibility

All components include:
- Proper ARIA labels
- Keyboard navigation support
- Focus management
- Screen reader compatibility

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

Proprietary - Blatam Academy



