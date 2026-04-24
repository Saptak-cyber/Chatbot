import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DocMind — PDF Conversational Agent',
  description:
    'Upload PDFs and chat with them using AI. Strictly grounded answers with page-level citations. Powered by Llama 3.3 70B via Groq.',
  keywords: ['PDF', 'AI', 'RAG', 'conversational agent', 'document Q&A'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📄</text></svg>" />
      </head>
      <body>{children}</body>
    </html>
  );
}
