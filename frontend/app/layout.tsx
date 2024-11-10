import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Toaster } from '@/components/ui/toaster';
import { ThemeProvider } from '@/components/providers/ThemeProvider';
import QueryProvider from '@/components/providers/query-provider';
import { LoadingProvider } from '@/components/providers/LoadingProvider';
import LoadingScreen from '@/components/common/Loader';
import Sidebar from '@/components/layouts/sidebar';
import Navbar from '@/components/layouts/navbar';
import { ChatProvider } from '@/components/providers/ChatProvider'; // Import ChatProvider

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AI Prompting',
  description: 'Life is never been easier using with AI Prompting',
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
  <ChatProvider>
    <Layout>{children}</Layout>
  </ChatProvider>
);

}

// Separate Layout component to manage Sidebar and Navbar
const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <html lang="en" suppressHydrationWarning={true}>
      <body className={inter.className}>
        <LoadingProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="light"
            enableSystem
            disableTransitionOnChange
            storageKey="app-theme"
          >
            <QueryProvider>
              <LoadingScreen />
              <div className="grid min-h-screen w-full md:grid-cols-[220px_1fr] lg:grid-cols-[280px_1fr]">
                <Sidebar /> 
                <div className="flex flex-col">
                  <Navbar />
                  <main className="flex flex-1 flex-col gap-4 lg:gap-6">
                    {children}
                  </main>
                </div>
              </div>
              <Toaster />
            </QueryProvider>
          </ThemeProvider>
        </LoadingProvider>
      </body>
    </html>
  );
};
