'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Analysis } from "@/types/medical"
import { formatDate } from "@/lib/utils"

interface AnalysisTrendsProps {
  analyses: Analysis[]
}

export function AnalysisTrends({ analyses }: AnalysisTrendsProps) {
  const [selectedMetric, setSelectedMetric] = useState<'confidence' | 'modelAccuracy'>('confidence')

  const chartData = analyses.map(analysis => ({
    date: formatDate(analysis.date),
    confidence: analysis.confidence * 100,
    modelAccuracy: analysis.modelAccuracy * 100,
  })).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Analysis Trends Over Time</CardTitle>
        <CardDescription>Visualize changes in analysis results</CardDescription>
        <Select
          value={selectedMetric}
          onValueChange={(value: 'confidence' | 'modelAccuracy') => setSelectedMetric(value)}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select metric" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="confidence">Confidence</SelectItem>
            <SelectItem value="modelAccuracy">Model Accuracy</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{
                top: 5,
                right: 30,
                left: 20,
                bottom: 5,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey={selectedMetric}
                stroke={selectedMetric === 'confidence' ? '#8884d8' : '#82ca9d'}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
