import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export default async function middleware(request: NextRequest) {
  const isOnDashboard = request.nextUrl.pathname.startsWith("/dashboard");
  const isOnApuntes = request.nextUrl.pathname.startsWith("/apuntes");
  const isOnLogin = request.nextUrl.pathname === "/login";
  const isOnApi = request.nextUrl.pathname.startsWith("/api");
  const isLoggedIn = request.cookies.has("next-auth.session-token");


  if (isOnLogin && isLoggedIn) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  if ((isOnDashboard || isOnApuntes) && !isLoggedIn) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/apuntes/:path*",
    "/login",
    "/api/:path*"
  ],
};         