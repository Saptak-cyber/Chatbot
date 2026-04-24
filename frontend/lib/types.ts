export interface PDFInfo {
  id: string;
  name: string;
  page_count: number;
  chunk_count: number;
}

export interface Citation {
  pdf_name: string;
  page_number: number;
  score?: number;  // cosine similarity (0–1) of the best matching chunk on this page
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources_used?: Citation[];
  is_grounded?: boolean;
  retrieval_score?: number;
  timestamp: Date;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  sources_used: Citation[];
  is_grounded: boolean;
  retrieval_score?: number;
}

export interface UploadResponse {
  id: string;
  name: string;
  page_count: number;
  chunk_count: number;
  message: string;
}
