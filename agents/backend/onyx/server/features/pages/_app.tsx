import { Toaster } from 'react-hot-toast';
import type { AppProps } from 'next/app';

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Toaster position="top-right" toastOptions={{ duration: 3500 }} />
      <Component {...pageProps} />
    </>
  );
}

export default MyApp; 