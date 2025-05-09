import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { GradCAMVisualization } from "./grad-cam-visualization"
import { Analysis } from "@/types/medical"

interface ComparisonViewProps {
  analyses: Analysis[]
}

export function ComparisonView({ analyses }: ComparisonViewProps) {
  const [selectedAnalyses, setSelectedAnalyses] = useState<string[]>([])

  const handleAnalysisSelect = (index: number) => (value: string) => {
    setSelectedAnalyses(prev => {
      const newSelected = [...prev]
      newSelected[index] = value
      return newSelected
    })
  }

  return (
    <Card className="w-full max-w-7xl mx-auto">
      <CardHeader>
        <CardTitle>Analysis Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[0, 1].map((index) => (
            <div key={index} className="space-y-4">
              <Select onValueChange={handleAnalysisSelect(index)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select an analysis" />
                </SelectTrigger>
                <SelectContent>
                  {analyses.map((analysis) => (
                    <SelectItem key={analysis.id} value={analysis.id}>
                      {analysis.type} - {new Date(analysis.date).toLocaleDateString()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedAnalyses[index] && (
                <GradCAMVisualization
                  analysisId={selectedAnalyses[index]}
                  originalImage={analyses.find(a => a.id === selectedAnalyses[index])?.visualization || ''}
                />
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

