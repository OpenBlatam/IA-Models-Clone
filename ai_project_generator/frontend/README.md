# AI Project Generator - Frontend

Next.js TypeScript frontend for the AI Project Generator API.

## Features

- 🚀 Generate AI projects with a user-friendly form
- 📊 Real-time project queue monitoring
- 📈 Statistics and analytics dashboard
- 🔄 WebSocket integration for live updates
- 📦 Export projects as ZIP/TAR
- 🎨 Modern UI with TailwindCSS

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running on `http://localhost:8020`

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## Environment Variables

Create a `.env.local` file:

```
NEXT_PUBLIC_API_URL=http://localhost:8020
NEXT_PUBLIC_WS_URL=ws://localhost:8020
```

## Project Structure

```
frontend/
├── app/              # Next.js app directory
│   ├── layout.tsx   # Root layout
│   ├── page.tsx     # Main dashboard page
│   └── globals.css  # Global styles
├── components/       # React components
│   ├── ProjectGeneratorForm.tsx
│   ├── ProjectQueue.tsx
│   ├── ProjectList.tsx
│   ├── Statistics.tsx
│   └── StatusIndicator.tsx
├── hooks/           # Custom React hooks
│   └── useWebSocket.ts
├── lib/             # Utilities and API client
│   └── api.ts
├── types/           # TypeScript types
│   └── index.ts
└── package.json
```

## Features

### Project Generation
- Detailed form with all project options
- Real-time validation
- Tag management
- Framework selection

### Queue Management
- View all queued projects
- Delete projects from queue
- Real-time updates

### Project List
- Filter by status (all, completed, failed, processing)
- Export projects
- View project details

### Statistics
- Total projects count
- Success/failure rates
- Average generation time
- Projects by type and framework

### Real-time Updates
- WebSocket connection for live updates
- Automatic reconnection
- Project-specific subscriptions

## Technologies

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **TailwindCSS** - Styling
- **Axios** - HTTP client
- **Lucide React** - Icons

## License

MIT

