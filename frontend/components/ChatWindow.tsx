'use client';

import {
  useEffect,
  useRef,
  useState,
  useCallback,
  KeyboardEvent,
} from 'react';
import {
  MessageSquare,
  Send,
  Trash2,
  AlertTriangle,
  Sparkles,
  FileText,
  Lock,
  Zap,
} from 'lucide-react';
import MessageBubble from './MessageBubble';
import LanguageSelector from './LanguageSelector';
import { Message, PDFInfo } from '@/lib/types';
import { clearChatHistory, sendMessageStream } from '@/lib/api';

interface ChatWindowProps {
  activePdfIds: string[];
  pdfs: PDFInfo[];
  sessionId: string;
}

export default function ChatWindow({
  activePdfIds,
  pdfs,
  sessionId,
}: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('auto');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const storageKey = `docmind_messages_${sessionId}`;
  const wordQueueRef = useRef<string[]>([]);
  const isStreamingRef = useRef(false);
  const currentMessageIdRef = useRef<string | null>(null);
  const hasLoadedFromStorage = useRef(false);

  const activePdfs = pdfs.filter((p) => activePdfIds.includes(p.id));
  const hasActivePdfs = activePdfIds.length > 0;
  const canSend = hasActivePdfs && input.trim().length > 0 && !isLoading;

  // Auto-focus textarea on any printable keypress when no other input is focused
  useEffect(() => {
    const handleGlobalKey = (e: globalThis.KeyboardEvent) => {
      // Ignore if already typing in an input, textarea, or contenteditable
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement).isContentEditable) return;
      // Ignore modifier-only combos (Ctrl+C, Cmd+R, etc.) and special keys
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key.length !== 1) return; // skip Enter, Backspace, ArrowKeys, F-keys…
      // Only focus if the textarea is enabled
      if (textareaRef.current && !textareaRef.current.disabled) {
        textareaRef.current.focus();
      }
    };
    window.addEventListener('keydown', handleGlobalKey);
    return () => window.removeEventListener('keydown', handleGlobalKey);
  }, []);

  // Restore messages from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as (Omit<Message, 'timestamp'> & { timestamp: string })[];
        setMessages(parsed.map((m) => ({ ...m, timestamp: new Date(m.timestamp) })));
      } catch {
        // Corrupted data — start fresh
      }
    }
    hasLoadedFromStorage.current = true;
  }, [storageKey]);

  // Persist messages to localStorage whenever they change
  useEffect(() => {
    if (!hasLoadedFromStorage.current) return;
    if (messages.length > 0) {
      localStorage.setItem(storageKey, JSON.stringify(messages));
    }
  }, [messages, storageKey]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Word-by-word streaming effect
  useEffect(() => {
    if (!isStreamingRef.current || wordQueueRef.current.length === 0) return;

    const timer = setInterval(() => {
      if (wordQueueRef.current.length === 0) {
        isStreamingRef.current = false;
        return;
      }

      const word = wordQueueRef.current.shift();
      if (word && currentMessageIdRef.current) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === currentMessageIdRef.current
              ? { ...msg, content: msg.content + word }
              : msg
          )
        );
      }
    }, 30); // Display one word every 30ms

    return () => clearInterval(timer);
  }, [messages]); // Re-run when messages change

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  };

  const handleSend = useCallback(async () => {
    if (!canSend) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    setIsLoading(true);

    // Create placeholder message
    const assistantId = `msg-${Date.now() + 1}`;
    currentMessageIdRef.current = assistantId;
    const placeholderMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, placeholderMessage]);

    // Reset word queue and start streaming
    wordQueueRef.current = [];
    isStreamingRef.current = true;

    try {
      await sendMessageStream(
        sessionId,
        userMessage.content,
        activePdfIds,
        {
          onChunk: (chunk: string) => {
            // Split chunk into words and add to queue
            const words = chunk.split(/(\s+)/); // Keep spaces
            wordQueueRef.current.push(...words);
            
            // Trigger re-render to start the word-by-word effect
            setMessages((prev) => [...prev]);
          },
          onMetadata: (data: any) => {
            // Optional: handle metadata if needed
            console.log('Metadata:', data);
          },
          onDone: (data: any) => {
            // Wait for queue to finish, then update metadata
            const checkQueue = setInterval(() => {
              if (wordQueueRef.current.length === 0) {
                clearInterval(checkQueue);
                isStreamingRef.current = false;
                currentMessageIdRef.current = null;
                
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantId
                      ? {
                          ...msg,
                          sources_used: data.sources || [],
                          is_grounded: data.is_grounded,
                          retrieval_score: data.retrieval_score,
                          confidence_level: data.confidence_level,
                          num_sources: data.num_sources,
                        }
                      : msg
                  )
                );
                setIsLoading(false);
              }
            }, 50);
          },
          onError: (error: string) => {
            wordQueueRef.current = [];
            isStreamingRef.current = false;
            currentMessageIdRef.current = null;
            
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, content: `⚠️ ${error}` }
                  : msg
              )
            );
            setIsLoading(false);
          },
        },
        selectedLanguage,
      );
    } catch (err) {
      wordQueueRef.current = [];
      isStreamingRef.current = false;
      currentMessageIdRef.current = null;
      
      const errorMsg =
        err instanceof Error ? err.message : 'Something went wrong. Please try again.';
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: `⚠️ ${errorMsg}` }
            : msg
        )
      );
      setIsLoading(false);
    }
  }, [canSend, input, sessionId, activePdfIds]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = async () => {
    if (messages.length === 0) return;
    if (!confirm('Clear conversation history?')) return;
    setMessages([]);
    localStorage.removeItem(storageKey);
    try {
      await clearChatHistory(sessionId);
    } catch {
      // Non-critical — UI already cleared
    }
  };

  return (
    <main className="chat-panel">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header-left">
          <div className="chat-title">
            <MessageSquare size={16} color="var(--accent-light)" />
            Chat
          </div>

          {hasActivePdfs ? (
            <div className="chat-active-pdfs">
              {activePdfs.map((pdf) => (
                <span key={pdf.id} className="active-pdf-chip" title={pdf.name}>
                  <span className="active-pdf-chip-dot" />
                  {pdf.name.replace('.pdf', '')}
                </span>
              ))}
            </div>
          ) : (
            <div className="no-pdfs-hint">
              <AlertTriangle size={11} />
              No PDFs loaded — upload &amp; load PDFs to start
            </div>
          )}
        </div>

        <div className="chat-header-right">
          <LanguageSelector
            selectedLanguage={selectedLanguage}
            onLanguageChange={setSelectedLanguage}
          />
          <button
            id="clear-chat-btn"
            className="btn-clear-chat"
            onClick={handleClearChat}
            disabled={messages.length === 0}
            title="Clear conversation history"
          >
            <Trash2 size={13} />
            Clear
          </button>
        </div>
      </div>

      {/* Messages area */}
      <div className="messages-container" id="messages-container">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-icon">
              <Sparkles size={32} color="var(--accent-light)" />
            </div>
            <h1>Ask your documents anything</h1>
            <p>
              Upload PDFs in the sidebar, select them, click{' '}
              <strong>Load Selected PDFs</strong>, then start asking questions.
            </p>
            <div className="chat-empty-hints">
              <div className="chat-empty-hint">
                <Lock size={14} className="hint-icon" />
                Answers are strictly grounded in your PDFs
              </div>
              <div className="chat-empty-hint">
                <FileText size={14} className="hint-icon" />
                Every response includes page number citations
              </div>
              <div className="chat-empty-hint">
                <Zap size={14} className="hint-icon" />
                Powered by Llama 3.3 70B via Groq
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              role={msg.role}
              content={msg.content}
              sources={msg.sources_used}
              isGrounded={msg.is_grounded}
              retrievalScore={msg.retrieval_score}
              confidenceLevel={msg.confidence_level}
              numSources={msg.num_sources}
              timestamp={msg.timestamp}
              isError={msg.content.startsWith('⚠️')}
            />
          ))
        )}

        {/* Typing indicator - only show if no placeholder message */}
        {isLoading && messages.filter(m => m.role === 'assistant' && !m.content).length === 0 && (
          <div className="message-wrapper assistant">
            <div
              className="message-meta"
              style={{ flexDirection: 'row' }}
            >
              <div className="message-avatar assistant">
                <FileText size={14} />
              </div>
            </div>
            <div className="typing-indicator">
              <div className="typing-dot" />
              <div className="typing-dot" />
              <div className="typing-dot" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            id="chat-input"
            className="chat-input"
            placeholder={
              hasActivePdfs
                ? 'Ask a question about your PDFs… (Enter to send, Shift+Enter for newline)'
                : 'Load PDFs from the sidebar first…'
            }
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={!hasActivePdfs || isLoading}
          />
          <button
            id="send-btn"
            className="btn-send"
            onClick={handleSend}
            disabled={!canSend}
            title="Send message"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="input-hint">
          Answers are strictly grounded in your selected PDFs •{' '}
          {hasActivePdfs
            ? `${activePdfIds.length} PDF${activePdfIds.length > 1 ? 's' : ''} active`
            : 'No PDFs loaded'}
        </p>
      </div>
    </main>
  );
}
