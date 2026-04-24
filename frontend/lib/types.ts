export interface PDFInfo {
  id: string;
  name: string;
  page_count: number;
  chunk_count: number;
}

export interface Citation {
  pdf_name: string;
  page_number: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources_used?: Citation[];
  timestamp: Date;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  sources_used: Citation[];
}

export interface UploadResponse {
  id: string;
  name: string;
  page_count: number;
  chunk_count: number;
  message: string;
}
