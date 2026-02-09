/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', 'sans-serif'],
        display: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      colors: {
        // Tesla Color Palette
        tesla: {
          black: '#171a20',
          'gray-dark': '#393c41',
          'gray-light': '#b5b5b5',
          white: '#ffffff',
          blue: '#0062cc',
        },
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0062cc', // Tesla Blue
          600: '#0052a3',
          700: '#004280',
          800: '#00325d',
          900: '#00223a',
        },
        robot: {
          dark: '#1a1a2e',
          medium: '#16213e',
          light: '#0f3460',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      boxShadow: {
        // Tesla exact shadows
        'tesla-xs': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'tesla': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        'tesla-sm': '0 2px 4px 0 rgba(0, 0, 0, 0.06), 0 1px 2px 0 rgba(0, 0, 0, 0.04)',
        'tesla-md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'tesla-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'tesla-xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        'tesla-2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        'tesla-inner': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
      },
      zIndex: {
        // Tesla exact z-index scale
        'tesla-dropdown': '1000',
        'tesla-sticky': '1020',
        'tesla-fixed': '1030',
        'tesla-modal-backdrop': '1040',
        'tesla-modal': '1050',
        'tesla-popover': '1060',
        'tesla-tooltip': '1070',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        // Tesla exact spacing scale in pixels
        'tesla-xs': '8px',
        'tesla-sm': '12px',
        'tesla-md': '16px',
        'tesla-lg': '24px',
        'tesla-xl': '32px',
        'tesla-2xl': '48px',
        'tesla-3xl': '64px',
        'tesla-4xl': '96px',
        'tesla-5xl': '128px',
      },
      padding: {
        // Tesla exact padding values
        'tesla-xs': '8px',
        'tesla-sm': '12px',
        'tesla-md': '16px',
        'tesla-lg': '24px',
        'tesla-xl': '32px',
        'tesla-2xl': '48px',
        'tesla-3xl': '64px',
        'tesla-4xl': '96px',
      },
      margin: {
        // Tesla exact margin values
        'tesla-xs': '8px',
        'tesla-sm': '12px',
        'tesla-md': '16px',
        'tesla-lg': '24px',
        'tesla-xl': '32px',
        'tesla-2xl': '48px',
        'tesla-3xl': '64px',
        'tesla-4xl': '96px',
      },
      letterSpacing: {
        'tesla': '-0.02em',
        'tesla-tight': '-0.04em',
        'tesla-wide': '0.02em',
      },
      fontSize: {
        // Tesla exact typography scale in pixels
        'tesla-xs': ['12px', { lineHeight: '1.5', letterSpacing: '0' }],
        'tesla-sm': ['14px', { lineHeight: '1.5', letterSpacing: '0' }],
        'tesla-base': ['16px', { lineHeight: '1.5', letterSpacing: '0' }],
        'tesla-lg': ['18px', { lineHeight: '1.5', letterSpacing: '0' }],
        'tesla-xl': ['20px', { lineHeight: '1.4', letterSpacing: '-0.01em' }],
        'tesla-2xl': ['24px', { lineHeight: '1.3', letterSpacing: '-0.02em' }],
        'tesla-3xl': ['30px', { lineHeight: '1.25', letterSpacing: '-0.02em' }],
        'tesla-4xl': ['36px', { lineHeight: '1.2', letterSpacing: '-0.02em' }],
        'tesla-5xl': ['48px', { lineHeight: '1.2', letterSpacing: '-0.02em' }],
        'tesla-6xl': ['60px', { lineHeight: '1.2', letterSpacing: '-0.02em' }],
        'tesla-7xl': ['72px', { lineHeight: '1.1', letterSpacing: '-0.04em' }],
        'tesla-8xl': ['96px', { lineHeight: '1.1', letterSpacing: '-0.04em' }],
        'tesla-hero': ['clamp(40px, 8vw, 96px)', { lineHeight: '1.1', letterSpacing: '-0.04em', fontWeight: '700' }],
        'tesla-display': ['clamp(30px, 5vw, 60px)', { lineHeight: '1.2', letterSpacing: '-0.02em', fontWeight: '600' }],
      },
      borderRadius: {
        // Tesla exact border radius
        'tesla-xs': '2px',
        'tesla-sm': '4px',
        'tesla-md': '6px',
        'tesla-lg': '8px',
        'tesla-xl': '12px',
        'tesla-2xl': '16px',
        'tesla-full': '9999px',
      },
      transitionDuration: {
        // Tesla exact transition timings
        'tesla-fast': '150ms',
        'tesla-base': '200ms',
        'tesla-slow': '300ms',
        'tesla-slower': '400ms',
      },
      transitionTimingFunction: {
        // Tesla exact easing curves
        'tesla-in': 'cubic-bezier(0.4, 0, 1, 1)',
        'tesla-out': 'cubic-bezier(0, 0, 0.2, 1)',
        'tesla-in-out': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'tesla-spring': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
      screens: {
        'xs': '475px',
        '3xl': '1920px',
      },
      minHeight: {
        'touch': '44px',
      },
      minWidth: {
        'touch': '44px',
      },
      opacity: {
        // Tesla exact opacity values
        '5': '0.05',
        '10': '0.1',
        '20': '0.2',
        '30': '0.3',
        '40': '0.4',
        '60': '0.6',
        '70': '0.7',
        '80': '0.8',
        '90': '0.9',
        '95': '0.95',
      },
      blur: {
        // Tesla exact blur values
        'tesla-sm': '4px',
        'tesla-base': '8px',
        'tesla-md': '12px',
        'tesla-lg': '16px',
        'tesla-xl': '20px',
        'tesla-2xl': '24px',
        'tesla-3xl': '40px',
      },
      backdropBlur: {
        // Tesla exact backdrop blur values
        'tesla-sm': '4px',
        'tesla-base': '8px',
        'tesla-md': '12px',
        'tesla-lg': '16px',
        'tesla-xl': '20px',
        'tesla-2xl': '24px',
        'tesla-3xl': '40px',
      },
      scale: {
        // Tesla exact scale values
        '95': '0.95',
        '98': '0.98',
        '102': '1.02',
        '105': '1.05',
        '110': '1.1',
      },
      translate: {
        // Tesla exact translate values
        'tesla-1': '-1px',
        'tesla-2': '-2px',
        'tesla-4': '-4px',
        'tesla-8': '-8px',
        'tesla-12': '-12px',
      },
      gap: {
        // Tesla exact gap values
        'tesla-xs': '8px',
        'tesla-sm': '12px',
        'tesla-md': '16px',
        'tesla-lg': '24px',
        'tesla-xl': '32px',
        'tesla-2xl': '48px',
      },
      padding: {
        // Tesla exact padding values
        'tesla-xs': '8px',
        'tesla-sm': '12px',
        'tesla-md': '16px',
        'tesla-lg': '24px',
        'tesla-xl': '32px',
        'tesla-2xl': '48px',
        'tesla-3xl': '64px',
        'tesla-4xl': '96px',
      },
      margin: {
        // Tesla exact margin values
        'tesla-xs': '8px',
        'tesla-sm': '12px',
        'tesla-md': '16px',
        'tesla-lg': '24px',
        'tesla-xl': '32px',
        'tesla-2xl': '48px',
        'tesla-3xl': '64px',
        'tesla-4xl': '96px',
      },
      borderWidth: {
        // Tesla exact border widths
        'tesla-1': '1px',
        'tesla-2': '2px',
        'tesla-4': '4px',
      },
    },
  },
  plugins: [],
}

