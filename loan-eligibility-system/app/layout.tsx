import './globals.css';
import { ReactNode } from 'react';

export const metadata = {
  title: 'Loan Eligibility System',
  description: 'Check your loan eligibility in real time',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900">{children}</body>
    </html>
  );
}
