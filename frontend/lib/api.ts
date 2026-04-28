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
  responseLanguage: string = 'auto',
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      active_pdf_ids: activePdfIds,
      response_language: responseLanguage,
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

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(10000), // 10 second timeout
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function sendMessageStream(
  sessionId: string,
  message: string,
  activePdfIds: string[],
  callbacks: {
    onChunk: (chunk: string) => void;
    onMetadata?: (data: any) => void;
    onDone: (data: any) => void;
    onError?: (error: string) => void;
  },
  responseLanguage: string = 'auto',
): Promise<void> {
  const res = await fetch(`${API_URL}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      active_pdf_ids: activePdfIds,
      response_language: responseLanguage,
    }),
  });

  if (!res.ok) {
    const error = `HTTP ${res.status}`;
    callbacks.onError?.(error);
    throw new Error(error);
  }

  const reader = res.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    callbacks.onError?.('No response body');
    throw new Error('No response body');
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
              case 'metadata':
                callbacks.onMetadata?.(data);
                break;
              case 'chunk':
                callbacks.onChunk(data.content);
                break;
              case 'done':
                callbacks.onDone(data);
                break;
              case 'refusal':
                callbacks.onChunk(data.content);
                callbacks.onDone({ is_grounded: false, sources: [] });
                break;
              case 'error':
                callbacks.onError?.(data.message);
                break;
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } catch (error) {
    callbacks.onError?.(error instanceof Error ? error.message : 'Stream error');
    throw error;
  }
}
