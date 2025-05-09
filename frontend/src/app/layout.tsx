import { Inter } from 'next/font/google'
import { MainLayout } from '@/components/layout/main-layout'
import '../styles/globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Medical Analysis System',
  description: 'Professional medical image analysis system for detecting brain tumors and cancer',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <MainLayout>
          {children}
        </MainLayout>
      </body>
    </html>
  )
}

