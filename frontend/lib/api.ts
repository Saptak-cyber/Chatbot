import { PDFInfo, ChatResponse, UploadResponse } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore parse errors
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function uploadPDF(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse<UploadResponse>(res);
}

export async function listPDFs(): Promise<PDFInfo[]> {
  const res = await fetch(`${API_URL}/api/pdfs`);
  return handleResponse<PDFInfo[]>(res);
}

export async function deletePDF(pdfId: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/pdfs/${pdfId}`, {
    method: 'DELETE',
  });
  await handleResponse<unknown>(res);
}

export async function sendMessage(
  sessionId: string,
  message: string,
  activePdfIds: string[],
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      active_pdf_ids: activePdfIds,
    }),
  });

  return handleResponse<ChatResponse>(res);
}

export async function clearChatHistory(sessionId: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/chat/${sessionId}`, {
    method: 'DELETE',
  });
  await handleResponse<unknown>(res);
}
