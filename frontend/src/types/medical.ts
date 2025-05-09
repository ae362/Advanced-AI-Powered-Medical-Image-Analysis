export interface Patient {
  id: string
  name: string
  dateOfBirth: string
  gender: 'male' | 'female' | 'other'
  email: string;
  contactDetails: string
  medicalHistory: string
  createdAt: string
}

export interface Analysis {
  id: string
  patientId: string
  date: string;
  type: 'brain_tumor' | 'cancer' // Updated to be specific
  prediction: 'positive' | 'negative'
  confidence: number
  visualization: string;
  createdAt: string
  modelAccuracy: number
  notes: Note[] | string | null;
  notificationEmail?: string
}

export interface AIAnalysisResponse {
  image_analysis: string;
  tumor_characteristics: string;
  stage: string;
  analysis: string;
  progression_prediction: string;
  recommended_actions: string;
}
export interface Note {
  x: number;
  y: number;
  text: string;
}

export interface AnalysisFormData {
  type: 'brain_tumor' | 'cancer'  // Keep this strict for now since these are the only supported types
  patientId: string;
  image: File;
}

export interface Disease {
  id: string;
  name: string;
  key: string;
  description: string;
  is_active: boolean;
  created_at: string;
}