'use client'

import { Document, Page, Text, View, StyleSheet, Image } from '@react-pdf/renderer'
import { Patient, Analysis } from '@/types/medical'

const styles = StyleSheet.create({
  page: {
    padding: 30,
    fontSize: 12,
  },
  header: {
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 18,
    marginBottom: 15,
    color: '#666',
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    marginBottom: 10,
    backgroundColor: '#f3f4f6',
    padding: 5,
  },
  row: {
    flexDirection: 'row',
    marginBottom: 5,
  },
  label: {
    width: 120,
    fontWeight: 'bold',
  },
  value: {
    flex: 1,
  },
  analysisImage: {
    width: '100%',
    height: 300,
    objectFit: 'contain',
    marginVertical: 10,
  },
  notes: {
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f9fafb',
  },
})

interface ReportPDFProps {
  patient: Patient
  analyses: Analysis[]
  doctorNotes: string
}

export function ReportPDF({ patient, analyses, doctorNotes }: ReportPDFProps) {
  return (
    <Document>
      <Page size="A4" style={styles.page}>
        <View style={styles.header}>
          <Text style={styles.title}>Medical Report</Text>
          <Text style={styles.subtitle}>Patient ID: {patient.id}</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Patient Information</Text>
          <View style={styles.row}>
            <Text style={styles.label}>Name:</Text>
            <Text style={styles.value}>{patient.name}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Date of Birth:</Text>
            <Text style={styles.value}>{new Date(patient.dateOfBirth).toLocaleDateString()}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Gender:</Text>
            <Text style={styles.value}>{patient.gender}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Contact:</Text>
            <Text style={styles.value}>{patient.contactDetails}</Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Analysis History</Text>
          {analyses.map((analysis, index) => (
            <View key={analysis.id} style={{ marginBottom: 20 }}>
              <Text style={{ fontWeight: 'bold', marginBottom: 5 }}>
                Analysis {index + 1} - {new Date(analysis.date).toLocaleDateString()}
              </Text>
              <View style={styles.row}>
                <Text style={styles.label}>Type:</Text>
                <Text style={styles.value}>
                  {analysis.type === 'brain_tumor' ? 'Brain Tumor Detection' : 'Cancer Detection'}
                </Text>
              </View>
              <View style={styles.row}>
                <Text style={styles.label}>Result:</Text>
                <Text style={styles.value}>
                  {analysis.prediction === 'positive' ? 'Abnormal' : 'Normal'}
                </Text>
              </View>
              <View style={styles.row}>
                <Text style={styles.label}>Confidence:</Text>
                <Text style={styles.value}>
                  {(analysis.confidence * 100).toFixed(2)}%
                </Text>
              </View>
              <Image
                src={`data:image/png;base64,${analysis.visualization}`}
                style={styles.analysisImage}
              />
              {analysis.notes && (
                <View style={styles.notes}>
                  <Text style={{ fontWeight: 'bold', marginBottom: 5 }}>Analysis Notes:</Text>
                  <Text>{typeof analysis.notes === 'string' ? analysis.notes : JSON.stringify(analysis.notes)}</Text>
                </View>
              )}
            </View>
          ))}
        </View>

        {doctorNotes && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Doctor's Notes</Text>
            <Text style={styles.notes}>{doctorNotes}</Text>
          </View>
        )}
      </Page>
    </Document>
  )
}

