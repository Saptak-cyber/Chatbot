'use client';

import { useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import PDFSidebar from '@/components/PDFSidebar';
import ChatWindow from '@/components/ChatWindow';
import { PDFInfo } from '@/lib/types';
import { listPDFs } from '@/lib/api';

const SESSION_KEY = 'docmind_session_id';

function getOrCreateSessionId(): string {
  if (typeof window === 'undefined') return uuidv4();
  const stored = sessionStorage.getItem(SESSION_KEY);
  if (stored) return stored;
  const id = uuidv4();
  sessionStorage.setItem(SESSION_KEY, id);
  return id;
}

export default function HomePage() {
  const [pdfs, setPdfs] = useState<PDFInfo[]>([]);
  const [selectedPdfIds, setSelectedPdfIds] = useState<Set<string>>(new Set());
  const [activePdfIds, setActivePdfIds] = useState<string[]>([]);
  const [sessionId] = useState<string>(getOrCreateSessionId);
  const [error, setError] = useState<string | null>(null);

  // Load existing PDFs on mount
  useEffect(() => {
    listPDFs()
      .then((data) => setPdfs(data))
      .catch(() => {
        // Backend might not be running locally — no-op
      });
  }, []);

  const handleLoadPdfs = useCallback((ids: string[]) => {
    setActivePdfIds(ids);
  }, []);

  const handleError = useCallback((msg: string) => {
    setError(msg);
    setTimeout(() => setError(null), 5000);
  }, []);

  return (
    <div className="app-container">
      <PDFSidebar
        pdfs={pdfs}
        selectedPdfIds={selectedPdfIds}
        activePdfIds={activePdfIds}
        onPdfsChange={setPdfs}
        onSelectionChange={setSelectedPdfIds}
        onLoadPdfs={handleLoadPdfs}
        onError={handleError}
      />
      <ChatWindow
        activePdfIds={activePdfIds}
        pdfs={pdfs}
        sessionId={sessionId}
      />

      {/* Global error toast */}
      {error && (
        <div
          className="upload-toast error"
          style={{ zIndex: 200 }}
          role="alert"
        >
          <span style={{ color: 'var(--error)', fontSize: 16 }}>✕</span>
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
