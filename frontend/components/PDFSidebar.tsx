'use client';

import { useState, useRef, useCallback } from 'react';
import {
  FileText,
  Upload,
  Trash2,
  BookOpen,
  CheckSquare,
  AlertCircle,
} from 'lucide-react';
import { PDFInfo } from '@/lib/types';
import { uploadPDF, deletePDF } from '@/lib/api';

interface PDFSidebarProps {
  pdfs: PDFInfo[];
  selectedPdfIds: Set<string>;
  activePdfIds: string[];
  onPdfsChange: (pdfs: PDFInfo[]) => void;
  onSelectionChange: (ids: Set<string>) => void;
  onLoadPdfs: (ids: string[]) => void;
  onError: (msg: string) => void;
}

type ToastState = {
  message: string;
  type: 'loading' | 'success' | 'error';
} | null;

export default function PDFSidebar({
  pdfs,
  selectedPdfIds,
  activePdfIds,
  onPdfsChange,
  onSelectionChange,
  onLoadPdfs,
  onError,
}: PDFSidebarProps) {
  const [toast, setToast] = useState<ToastState>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const showToast = (message: string, type: ToastState['type']) => {
    setToast({ message, type });
    if (type !== 'loading') {
      setTimeout(() => setToast(null), 3500);
    }
  };

  const handleUploadClick = () => fileInputRef.current?.click();

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      // Reset input so same file can be re-uploaded
      e.target.value = '';

      showToast(`Processing "${file.name}"...`, 'loading');

      try {
        const result = await uploadPDF(file);
        const newPdf: PDFInfo = {
          id: result.id,
          name: result.name,
          page_count: result.page_count,
          chunk_count: result.chunk_count,
        };

        onPdfsChange([...pdfs, newPdf]);
        // Auto-select newly uploaded PDF
        const newSelected = new Set(selectedPdfIds);
        newSelected.add(newPdf.id);
        onSelectionChange(newSelected);

        showToast(
          `✓ "${result.name}" — ${result.chunk_count} chunks ready`,
          'success',
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Upload failed';
        showToast(msg, 'error');
        onError(msg);
      }
    },
    [pdfs, selectedPdfIds, onPdfsChange, onSelectionChange, onError],
  );

  const handleToggleSelect = (id: string) => {
    const updated = new Set(selectedPdfIds);
    if (updated.has(id)) {
      updated.delete(id);
    } else {
      updated.add(id);
    }
    onSelectionChange(updated);
  };

  const handleDelete = async (pdf: PDFInfo) => {
    if (!confirm(`Delete "${pdf.name}"? This cannot be undone.`)) return;

    setDeletingId(pdf.id);
    try {
      await deletePDF(pdf.id);
      const updated = pdfs.filter((p) => p.id !== pdf.id);
      onPdfsChange(updated);

      const newSelected = new Set(selectedPdfIds);
      newSelected.delete(pdf.id);
      onSelectionChange(newSelected);

      // If the deleted PDF was active, reload without it
      if (activePdfIds.includes(pdf.id)) {
        onLoadPdfs(activePdfIds.filter((id) => id !== pdf.id));
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Delete failed';
      onError(msg);
    } finally {
      setDeletingId(null);
    }
  };

  const handleLoadPdfs = () => {
    const ids = Array.from(selectedPdfIds);
    onLoadPdfs(ids);
  };

  const selectedCount = selectedPdfIds.size;
  const isLoaded =
    activePdfIds.length > 0 &&
    activePdfIds.length === selectedCount &&
    activePdfIds.every((id) => selectedPdfIds.has(id));

  return (
    <aside className="sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <div className="brand-icon">
            <FileText size={18} color="white" />
          </div>
          <div>
            <div className="brand-title">DocMind</div>
            <div className="brand-subtitle">PDF Agent</div>
          </div>
        </div>

        {/* Upload button */}
        <button
          id="upload-pdf-btn"
          className="btn-upload"
          onClick={handleUploadClick}
          disabled={toast?.type === 'loading'}
        >
          <Upload size={14} />
          Upload PDF
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id="pdf-file-input"
        />
      </div>

      {/* PDF List */}
      <div className="pdf-list-container">
        {pdfs.length === 0 ? (
          <div className="pdf-list-empty">
            <FileText size={40} className="pdf-list-empty-icon" />
            <p>
              No PDFs uploaded yet.
              <br />
              Upload a PDF to start chatting with it.
            </p>
          </div>
        ) : (
          <>
            <div className="pdf-list-title">
              {pdfs.length} Document{pdfs.length !== 1 ? 's' : ''}
            </div>
            {pdfs.map((pdf) => (
              <div
                key={pdf.id}
                className={`pdf-card ${selectedPdfIds.has(pdf.id) ? 'selected' : ''}`}
              >
                <input
                  type="checkbox"
                  className="pdf-checkbox"
                  checked={selectedPdfIds.has(pdf.id)}
                  onChange={() => handleToggleSelect(pdf.id)}
                  id={`pdf-check-${pdf.id}`}
                  title={`Select ${pdf.name}`}
                />
                <FileText size={14} className="pdf-card-icon" />
                <div className="pdf-card-info">
                  <div className="pdf-card-name" title={pdf.name}>
                    {pdf.name}
                  </div>
                  <div className="pdf-card-meta">
                    <span className="pdf-meta-badge">
                      <BookOpen size={9} />
                      {pdf.page_count}p
                    </span>
                    <span className="pdf-meta-badge">
                      <CheckSquare size={9} />
                      {pdf.chunk_count} chunks
                    </span>
                    {activePdfIds.includes(pdf.id) && (
                      <span
                        className="pdf-meta-badge"
                        style={{
                          color: 'var(--success)',
                          borderColor: 'rgba(16, 185, 129, 0.3)',
                          background: 'var(--success-dim)',
                        }}
                      >
                        Active
                      </span>
                    )}
                  </div>
                </div>
                <button
                  className="pdf-delete-btn"
                  onClick={() => handleDelete(pdf)}
                  disabled={deletingId === pdf.id}
                  title={`Delete ${pdf.name}`}
                  id={`delete-pdf-${pdf.id}`}
                >
                  {deletingId === pdf.id ? (
                    <div
                      className="spinner"
                      style={{ width: 12, height: 12 }}
                    />
                  ) : (
                    <Trash2 size={13} />
                  )}
                </button>
              </div>
            ))}
          </>
        )}
      </div>

      {/* Footer — Load PDFs button */}
      <div className="sidebar-footer">
        {selectedCount === 0 && pdfs.length > 0 && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 12,
              color: 'var(--warning)',
              marginBottom: 10,
            }}
          >
            <AlertCircle size={12} />
            Select PDFs to load them for chat
          </div>
        )}
        <button
          id="load-pdfs-btn"
          className="btn-load"
          onClick={handleLoadPdfs}
          disabled={selectedCount === 0}
        >
          {isLoaded ? (
            <>
              <CheckSquare size={14} />
              Loaded ({selectedCount})
            </>
          ) : (
            <>
              Load Selected PDFs
              {selectedCount > 0 && (
                <span className="load-count-badge">{selectedCount}</span>
              )}
            </>
          )}
        </button>
      </div>

      {/* Toast notification */}
      {toast && (
        <div className={`upload-toast ${toast.type !== 'loading' ? toast.type : ''}`}>
          {toast.type === 'loading' ? (
            <div className="spinner" />
          ) : toast.type === 'success' ? (
            <span style={{ color: 'var(--success)', fontSize: 16 }}>✓</span>
          ) : (
            <AlertCircle size={16} color="var(--error)" />
          )}
          <span>{toast.message}</span>
        </div>
      )}
    </aside>
  );
}
