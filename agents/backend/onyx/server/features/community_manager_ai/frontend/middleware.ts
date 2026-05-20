import createMiddleware from 'next-intl/middleware';
import { withAuth } from 'next-auth/middleware';
import { routing } from './i18n/routing';

const intlMiddleware = createMiddleware(routing);

const publicPages = ['/login'];

export default withAuth(
  function onSuccess(req) {
    return intlMiddleware(req);
  },
  {
    callbacks: {
      authorized: ({ token, req }) => {
        const { pathname } = req.nextUrl;
        const isPublicPage = publicPages.some((page) => 
          pathname.includes(page) || pathname === '/'
        );
        
        if (isPublicPage) {
          return true;
        }

        return !!token;
      },
    },
    pages: {
      signIn: '/login',
    },
  }
);

export const config = {
  matcher: [
    '/((?!api|_next|_vercel|.*\\..*).*)',
    '/(es|en|fr|pt)/:path*',
  ],
};
