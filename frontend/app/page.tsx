'use client';

import { useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import PDFSidebar from '@/components/PDFSidebar';
import ChatWindow from '@/components/ChatWindow';
import ColdStartBanner from '@/components/ColdStartBanner';
import { PDFInfo, ConversationThread } from '@/lib/types';
import { listPDFs } from '@/lib/api';

const CONVERSATIONS_KEY = 'docmind_conversations';
const ACTIVE_SESSION_KEY = 'docmind_active_session';

function loadThreads(): ConversationThread[] {
  if (typeof window === 'undefined') return [];
  try {
    const s = localStorage.getItem(CONVERSATIONS_KEY);
    return s ? JSON.parse(s) : [];
  } catch { return []; }
}

function saveThreads(threads: ConversationThread[]) {
  if (typeof window === 'undefined') return;
  localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(threads));
}

function makeThread(activePdfIds: string[] = []): ConversationThread {
  return {
    id: uuidv4(),
    title: 'New conversation',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    activePdfIds,
    previewText: '',
    messageCount: 0,
  };
}

export default function HomePage() {
  const [pdfs, setPdfs] = useState<PDFInfo[]>([]);
  const [selectedPdfIds, setSelectedPdfIds] = useState<Set<string>>(new Set());
  const [activePdfIds, setActivePdfIds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationThread[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>('');

  // Bootstrap from localStorage (client only)
  useEffect(() => {
    let threads = loadThreads();
    let activeId = localStorage.getItem(ACTIVE_SESSION_KEY) || '';

    if (threads.length === 0) {
      const t = makeThread();
      threads = [t];
      activeId = t.id;
      saveThreads(threads);
    } else if (!activeId || !threads.find((t) => t.id === activeId)) {
      activeId = threads[0].id;
    }

    localStorage.setItem(ACTIVE_SESSION_KEY, activeId);
    setConversations(threads);
    setActiveSessionId(activeId);

    const active = threads.find((t) => t.id === activeId);
    if (active?.activePdfIds.length) {
      setActivePdfIds(active.activePdfIds);
      setSelectedPdfIds(new Set(active.activePdfIds));
    }
  }, []);

  // Load PDFs from backend and clean up any orphaned IDs from localStorage
  useEffect(() => {
    listPDFs()
      .then((loadedPdfs) => {
        setPdfs(loadedPdfs);
        const validIds = new Set(loadedPdfs.map((p) => p.id));

        setActivePdfIds((prev) => {
          const cleaned = prev.filter((id) => validIds.has(id));
          if (cleaned.length !== prev.length) {
            // Update the active session in localStorage if we removed phantom IDs
            setConversations((threads) => {
              const updated = threads.map((c) =>
                c.id === activeSessionId ? { ...c, activePdfIds: cleaned } : c
              );
              saveThreads(updated);
              return updated;
            });
          }
          return cleaned;
        });

        setSelectedPdfIds((prev) => {
          // If no PDFs are currently selected and no active ones exist, select all by default
          if (prev.size === 0 && activePdfIds.length === 0 && loadedPdfs.length > 0) {
            return new Set(loadedPdfs.map((p) => p.id));
          }

          const next = new Set(prev);
          next.forEach((id) => {
            if (!validIds.has(id)) next.delete(id);
          });
          return next;
        });
      })
      .catch(() => {});
  }, [activeSessionId]);

  // ── Conversation actions ───────────────────────────────────────────────────

  const handleNewConversation = useCallback(() => {
    const t = makeThread(activePdfIds);
    setConversations((prev) => {
      const updated = [t, ...prev];
      saveThreads(updated);
      return updated;
    });
    setActiveSessionId(t.id);
    localStorage.setItem(ACTIVE_SESSION_KEY, t.id);
  }, [activePdfIds]);

  const handleSwitchConversation = useCallback((id: string) => {
    setConversations((prev) => {
      const t = prev.find((c) => c.id === id);
      if (!t) return prev;
      setActiveSessionId(id);
      localStorage.setItem(ACTIVE_SESSION_KEY, id);
      setActivePdfIds(t.activePdfIds);
      setSelectedPdfIds(new Set(t.activePdfIds));
      return prev;
    });
  }, []);

  const handleRenameConversation = useCallback((id: string, title: string) => {
    setConversations((prev) => {
      const updated = prev.map((c) =>
        c.id === id ? { ...c, title, updatedAt: Date.now() } : c
      );
      saveThreads(updated);
      return updated;
    });
  }, []);

  const handleDeleteConversation = useCallback(
    (id: string) => {
      localStorage.removeItem(`docmind_messages_${id}`);
      setConversations((prev) => {
        let updated = prev.filter((c) => c.id !== id);
        if (updated.length === 0) {
          updated = [makeThread(activePdfIds)];
        }
        saveThreads(updated);

        if (id === activeSessionId) {
          const newId = updated[0].id;
          setActiveSessionId(newId);
          localStorage.setItem(ACTIVE_SESSION_KEY, newId);
          const newActive = updated.find((c) => c.id === newId);
          setActivePdfIds(newActive?.activePdfIds ?? []);
          setSelectedPdfIds(new Set(newActive?.activePdfIds ?? []));
        }

        return updated;
      });
    },
    [activeSessionId, activePdfIds]
  );

  const handleThreadUpdate = useCallback(
    (update: { title?: string; preview?: string; messageCount?: number }) => {
      setConversations((prev) => {
        const updated = prev.map((c) => {
          if (c.id !== activeSessionId) return c;
          return {
            ...c,
            ...(update.title ? { title: update.title } : {}),
            ...(update.preview !== undefined ? { previewText: update.preview } : {}),
            ...(update.messageCount !== undefined ? { messageCount: update.messageCount } : {}),
            updatedAt: Date.now(),
          };
        });
        saveThreads(updated);
        return updated;
      });
    },
    [activeSessionId]
  );

  // ── PDF actions ────────────────────────────────────────────────────────────

  const handleLoadPdfs = useCallback(
    (ids: string[]) => {
      setActivePdfIds(ids);
      setConversations((prev) => {
        const updated = prev.map((c) =>
          c.id === activeSessionId ? { ...c, activePdfIds: ids, updatedAt: Date.now() } : c
        );
        saveThreads(updated);
        return updated;
      });
    },
    [activeSessionId]
  );

  const handleError = useCallback((msg: string) => {
    setError(msg);
    setTimeout(() => setError(null), 5000);
  }, []);

  return (
    <div className="app-container">
      <ColdStartBanner />
      <PDFSidebar
        pdfs={pdfs}
        selectedPdfIds={selectedPdfIds}
        activePdfIds={activePdfIds}
        onPdfsChange={setPdfs}
        onSelectionChange={setSelectedPdfIds}
        onLoadPdfs={handleLoadPdfs}
        onError={handleError}
        conversations={conversations}
        activeSessionId={activeSessionId}
        onNewConversation={handleNewConversation}
        onSwitchConversation={handleSwitchConversation}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={handleDeleteConversation}
      />
      {activeSessionId && (
        <ChatWindow
          activePdfIds={activePdfIds}
          pdfs={pdfs}
          sessionId={activeSessionId}
          onThreadUpdate={handleThreadUpdate}
        />
      )}

      {error && (
        <div className="upload-toast error" style={{ zIndex: 200 }} role="alert">
          <span style={{ color: 'var(--error)', fontSize: 16 }}>✕</span>
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
