import { Patient, Analysis,Disease,AnalysisFormData} from "@/types/medical"
import { AppError, handleError, retryOperation } from './error-handling';
const API_BASE_URL = 'http://localhost:8000/api'  // Update this to match your Django backend URL
import { getAccessToken, isTokenExpired, refreshAccessToken } from './auth';

async function fetchWithToken(url: string, options: RequestInit = {}) {
  let accessToken = getAccessToken();

  if (!accessToken || isTokenExpired(accessToken)) {
    try {
      accessToken = await refreshAccessToken();
    } catch (error) {
      throw new Error('Session expired. Please log in again.');
    }
  }

  const response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error('API request failed');
  }

  return response.json();
}
export async function getPatients(): Promise<Patient[]> {
  const response = await fetch(`${API_BASE_URL}/patients/`)
  if (!response.ok) {
    throw new Error('Failed to fetch patients')
  }
  const data = await response.json()
  return data.map((patient: any) => ({
    ...patient,
    dateOfBirth: patient.date_of_birth,
    contactDetails: patient.contact_details,
    medicalHistory: patient.medical_history,
    createdAt: patient.created_at,
  }))
}

export async function getPatient(id: string): Promise<Patient> {
  const response = await fetch(`${API_BASE_URL}/patients/${id}/`)
  if (!response.ok) {
    throw new Error('Failed to fetch patient')
  }
  const data = await response.json()
  return {
    ...data,
    dateOfBirth: data.date_of_birth,
    contactDetails: data.contact_details,
    medicalHistory: data.medical_history,
    createdAt: data.created_at,
  }
}

export async function createPatient(data: Omit<Patient, 'id' | 'createdAt'>): Promise<Patient> {
  const response = await fetch(`${API_BASE_URL}/patients/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name: data.name,
      email: data.email,
      date_of_birth: data.dateOfBirth,
      gender: data.gender,
      contact_details: data.contactDetails,
      medical_history: data.medicalHistory,
    }),
  })
  if (!response.ok) {
    throw new Error('Failed to create patient')
  }
  const responseData = await response.json()
  return {
    ...responseData,
    dateOfBirth: responseData.date_of_birth,
    contactDetails: responseData.contact_details,
    medicalHistory: responseData.medical_history,
    createdAt: responseData.created_at,
  }
}

export async function updatePatient(id: string, data: Partial<Patient>): Promise<Patient> {
  const response = await fetch(`${API_BASE_URL}/patients/${id}/`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ...(data.name && { name: data.name }),
      ...(data.dateOfBirth && { date_of_birth: data.dateOfBirth }),
      ...(data.gender && { gender: data.gender }),
      ...(data.contactDetails && { contact_details: data.contactDetails }),
      ...(data.medicalHistory && { medical_history: data.medicalHistory }),
    }),
  })
  if (!response.ok) {
    throw new Error('Failed to update patient')
  }
  const responseData = await response.json()
  return {
    ...responseData,
    dateOfBirth: responseData.date_of_birth,
    contactDetails: responseData.contact_details,
    medicalHistory: responseData.medical_history,
    createdAt: responseData.created_at,
  }
}

export async function deletePatient(id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/patients/${id}/`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    throw new Error('Failed to delete patient')
  }
}

export async function getPatientAnalyses(patientId: string): Promise<Analysis[]> {
  const response = await fetch(`${API_BASE_URL}/patients/${patientId}/analyses/`)
  if (!response.ok) {
    throw new Error('Failed to fetch analyses')
  }
  const data = await response.json()
  return data.map((analysis: any) => ({
    ...analysis,
    createdAt: analysis.created_at,
    modelAccuracy: analysis.model_accuracy,
  }))
}

export async function runAnalysis(data: FormData): Promise<Analysis> {
  const response = await fetch(`${API_BASE_URL}/analyses/predict/`, {
    method: 'POST',
    body: data,
  })
  if (!response.ok) {
    throw new Error('Failed to run analysis')
  }
  const responseData = await response.json()
  return {
    ...responseData,
    createdAt: responseData.created_at,
    modelAccuracy: responseData.model_accuracy,
  }
}
export async function saveAnalysisNote(analysisId: string, notes: string): Promise<Analysis> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/notes/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ notes }),
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to save analysis notes');
  }
  
  return response.json();
}
export async function deleteAnalysisNotes(analysisId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/notes/`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error('Failed to delete analysis notes');
  }
}

export async function deleteAnalysis(analysisId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to delete analysis');
  }
}
export async function addDisease(disease: Omit<Disease, "id" | "is_active" | "created_at">) {
  const response = await fetch('/api/diseases/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(disease),
  })

  if (!response.ok) {
    throw new Error('Failed to add disease')
  }

  return response.json()
}

export async function getDiseases() {
  const response = await fetch('/api/diseases/')
  
  if (!response.ok) {
    throw new Error('Failed to fetch diseases')
  }

  return response.json()
}
export async function getGradCAMVisualization(analysisId: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/gradcam/`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error('Failed to generate Grad-CAM visualization');
  }
  const data = await response.json();
  return data.gradcam;
}
export async function getAIExplanationAndStaging(analysisId: string): Promise<{
  image_analysis: string;
  tumor_characteristics: string;
  stage: string;
  progression_prediction: string;
  recommended_actions: string;
}> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/explanation-and-staging/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to generate AI explanation and staging');
  }

  return response.json();
}


export async function uploadTrainingImages(formData: FormData): Promise<{ paths: string[]; message: string; errors?: string[] }> {
  try {
    const response = await fetch(`${API_BASE_URL}/training/upload-images/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Invalid server response' }));
      throw new Error(errorData.error || 'Failed to upload images');
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error uploading images:', error);
    throw error;
  }
}



export async function startModelTraining({ 
  diseaseName, 
  classes 
}: { 
  diseaseName: string; 
  classes: { key: string; name: string }[] 
}): Promise<{ accuracy: number; confidence: number }> {
  try {
    const response = await retryOperation(() =>
      fetch(`${API_BASE_URL}/training/start/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          disease_name: diseaseName,
          training_data: classes
        }),
      })
    )

    // Add specific handling for 404
    if (response.status === 404) {
      throw new AppError('Training endpoint not found. Please check API configuration.', 'ENDPOINT_NOT_FOUND')
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Invalid server response' }))
      throw new AppError(errorData.error || 'Failed to start model training', 'TRAINING_FAILED')
    }

    const result = await response.json()
    return {
      accuracy: result.accuracy,
      confidence: result.confidence,
    }
  } catch (error) {
    handleError(error)
    throw error
  }
}


export async function updateNotificationEmail(analysisId: string, email: string): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/update-email/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email }),
  });

  if (!response.ok) {
    // Try to parse error as JSON, fall back to text if not JSON
    let errorMessage: string;
    try {
      const errorData = await response.json();
      errorMessage = errorData.error || 'Failed to update notification email';
    } catch {
      // If response is not JSON, get the text content
      const errorText = await response.text();
      errorMessage = 'Failed to update notification email. Server response was not valid JSON.';
      console.error('Server response:', errorText);
    }
    throw new Error(errorMessage);
  }

  return response.json();
}


export async function getAIAnalysis(analysisId: string): Promise<{
  analysis: string;
}> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/explanation-and-staging/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to generate AI analysis');
  }

  return response.json();
}
export async function submitAnalysis(data: AnalysisFormData): Promise<Analysis> {
  const formData = new FormData();
  formData.append('patientId', data.patientId);
  formData.append('type', data.type);
  formData.append('image', data.image);

  const response = await fetch(`${API_BASE_URL}/submit-analysis/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to submit analysis');
  }

  const analysisData = await response.json();
  return analysisData as Analysis;
}
export async function sendAnalysisEmail(analysisId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/send-email/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to send analysis email');
  }
}

export async function sendAnalysisToEmail(analysisId: string, email: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/analyses/${analysisId}/send-to-email/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to send analysis email');
  }
}