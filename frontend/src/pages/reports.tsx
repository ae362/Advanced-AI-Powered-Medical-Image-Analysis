'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { type Analysis, type Patient } from "@/types/medical"
import { formatDate } from "@/lib/utils"

// Mock data - replace with actual API calls
const mockReports: (Analysis & { patient: Patient })[] = [
  {
    id: '1',
    patientId: '1',
    patient: {
      id: '1',
      name: 'John Doe',
      dateOfBirth: '1980-05-15',
      gender: 'male',
      contactDetails: '+1 234 567 8900',
      medicalHistory: 'No significant medical history.',
      createdAt: '2023-01-15T09:00:00Z'
    },
    type: 'brain_tumor',
    prediction: 'negative',
    confidence: 0.95,
    visualization: '',
    createdAt: '2023-03-10T11:20:00Z',
    modelAccuracy: 0.92
  },
  // Add more mock reports as needed
]

export default function ReportsPage() {
  const [selectedReport, setSelectedReport] = useState<(typeof mockReports)[0] | null>(null)

  const handleViewReport = (report: (typeof mockReports)[0]) => {
    setSelectedReport(report)
  }

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Reports</h1>
        {selectedReport && (
          <Button variant="outline" onClick={() => setSelectedReport(null)}>
            Back to List
          </Button>
        )}
      </div>

      {selectedReport ? (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Report</CardTitle>
            <CardDescription>
              Patient: {selectedReport.patient.name} | Date: {formatDate(selectedReport.createdAt)}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-4">
              <div className="grid gap-2">
                <div className="font-medium">Analysis Type</div>
                <div className="text-sm">
                  {selectedReport.type === 'brain_tumor' ? 'Brain Tumor Analysis' : 'Cancer Analysis'}
                </div>
              </div>
              <div className="grid gap-2">
                <div className="font-medium">Result</div>
                <div className={`text-sm ${
                  selectedReport.prediction === 'positive' ? 'text-red-500' : 'text-green-500'
                }`}>
                  {selectedReport.prediction}
                </div>
              </div>
              <div className="grid gap-2">
                <div className="font-medium">Confidence</div>
                <div className="text-sm">
                  {(selectedReport.confidence * 100).toFixed(2)}%
                </div>
              </div>
              <div className="grid gap-2">
                <div className="font-medium">Model Accuracy</div>
                <div className="text-sm">
                  {(selectedReport.modelAccuracy * 100).toFixed(2)}%
                </div>
              </div>
            </div>

            {selectedReport.visualization && (
              <div className="space-y-2">
                <div className="font-medium">Visualization</div>
                <img
                  src={`data:image/png;base64,${selectedReport.visualization}`}
                  alt="Analysis Visualization"
                  className="w-full rounded-lg border"
                />
              </div>
            )}
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>All Reports</CardTitle>
            <CardDescription>View and manage analysis reports</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Patient</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Result</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {mockReports.map((report) => (
                  <TableRow key={report.id}>
                    <TableCell>{report.patient.name}</TableCell>
                    <TableCell>
                      {report.type === 'brain_tumor' ? 'Brain Tumor' : 'Cancer'}
                    </TableCell>
                    <TableCell>
                      <span className={
                        report.prediction === 'positive' ? 'text-red-500' : 'text-green-500'
                      }>
                        {report.prediction}
                      </span>
                    </TableCell>
                    <TableCell>{formatDate(report.createdAt)}</TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleViewReport(report)}
                      >
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

