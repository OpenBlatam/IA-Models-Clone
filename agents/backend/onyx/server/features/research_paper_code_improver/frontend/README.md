# Research Paper Code Improver - Frontend

Modern React/Next.js frontend for the Research Paper Code Improver system.

## 🚀 Features

- **Paper Management**: Upload PDFs or process papers from URLs
- **Model Training**: Train AI models using research papers
- **Code Improvement**: Improve code using knowledge from papers
- **Dashboard**: Real-time statistics and system monitoring
- **Responsive Design**: Mobile-first, fully responsive UI
- **Type Safety**: Full TypeScript coverage
- **Modern UI**: Built with TailwindCSS and modern design patterns

## 📋 Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running on `http://localhost:8030` (or configure via environment variables)

## 🛠️ Installation

```bash
cd frontend
npm install
```

## 🏃 Development

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## 🏗️ Build

```bash
npm run build
npm start
```

## 📁 Project Structure

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Dashboard page
│   ├── papers/             # Papers management
│   ├── training/           # Model training
│   ├── code-improve/       # Code improvement
│   └── globals.css         # Global styles
│
├── components/
│   ├── ui/                 # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   ├── Card.tsx
│   │   ├── Modal.tsx
│   │   └── ...
│   ├── features/           # Feature-specific components
│   │   ├── PaperUpload.tsx
│   │   ├── PaperList.tsx
│   │   ├── TrainingForm.tsx
│   │   └── CodeImprover.tsx
│   └── layout/             # Layout components
│       ├── Navbar.tsx
│       └── PageLayout.tsx
│
├── lib/
│   └── api/                # API client
│       ├── client.ts       # Axios client
│       ├── types.ts        # TypeScript types
│       └── index.ts
│
└── package.json
```

## 🎨 UI Components

### Base Components
- **Button**: Multiple variants (primary, secondary, outline, ghost, danger)
- **Input**: Text input with label, error, and helper text
- **Textarea**: Multi-line text input
- **Card**: Container with optional header/footer
- **Modal**: Accessible modal dialog
- **LoadingSpinner**: Loading indicator
- **Badge**: Status badges with variants

### Feature Components
- **PaperUpload**: Upload PDFs or process URLs
- **PaperList**: Display grid of papers
- **TrainingForm**: Configure and start model training
- **CodeImprover**: Improve code from GitHub or text

## 🔌 API Integration

The frontend uses a centralized API client (`lib/api/client.ts`) that:
- Handles all API calls to the backend
- Provides TypeScript type safety
- Includes error handling
- Supports file uploads

## 🎯 Key Features

### Paper Management
- Upload PDF files via drag-and-drop
- Process papers from URLs (arXiv, etc.)
- View all uploaded papers
- Paper details and metadata

### Model Training
- Select papers for training
- Configure training parameters (epochs, model name)
- Monitor training status
- View training results

### Code Improvement
- Improve code from GitHub repositories
- Improve code from text input
- Side-by-side comparison view
- Suggestions and recommendations
- Syntax highlighting

### Dashboard
- Real-time statistics
- System health monitoring
- Feature availability status
- Quick actions

## 🎨 Styling

- **TailwindCSS**: Utility-first CSS framework
- **Custom Colors**: Primary color scheme defined in `tailwind.config.ts`
- **Responsive**: Mobile-first design
- **Accessibility**: ARIA labels, keyboard navigation, focus states

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8030
```

### API Proxy

The `next.config.js` includes a rewrite rule to proxy API requests to the backend.

## 📱 Responsive Design

- Mobile-first approach
- Breakpoints: sm (640px), md (768px), lg (1024px), xl (1280px)
- Touch-friendly interactions
- Mobile navigation menu

## ♿ Accessibility

- ARIA labels on interactive elements
- Keyboard navigation support
- Focus states for all interactive elements
- Screen reader friendly
- Semantic HTML

## 🧪 Type Safety

- Full TypeScript coverage
- Types match backend API schemas
- Type-safe API client
- No `any` types in components

## 📦 Dependencies

### Core
- Next.js 14 (App Router)
- React 18
- TypeScript 5

### UI & Styling
- TailwindCSS
- Lucide React (icons)
- React Hot Toast (notifications)

### Data Fetching
- TanStack Query (React Query)
- Axios

### Code Highlighting
- React Syntax Highlighter

### File Upload
- React Dropzone

## 🚀 Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import project in Vercel
3. Set environment variables
4. Deploy

### Other Platforms

Build the application and serve the `out` directory:

```bash
npm run build
```

## 📝 Development Guidelines

- Use TypeScript for all new files
- Follow the component structure (UI components in `components/ui/`)
- Use TailwindCSS for styling (no CSS files)
- Implement accessibility features
- Use early returns for readability
- Name event handlers with "handle" prefix
- Use const arrow functions instead of function declarations

## 🐛 Troubleshooting

### API Connection Issues
- Verify backend is running on port 8030
- Check `NEXT_PUBLIC_API_URL` environment variable
- Check browser console for CORS errors

### Build Errors
- Run `npm run type-check` to check TypeScript errors
- Clear `.next` directory and rebuild
- Verify all dependencies are installed

## 📚 Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [TailwindCSS Documentation](https://tailwindcss.com/docs)
- [TanStack Query Documentation](https://tanstack.com/query/latest)
- [React Documentation](https://react.dev)




