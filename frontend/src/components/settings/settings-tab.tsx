'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { AddDiseaseDialog } from "./add-disease-dialog"
import { UploadModelDialog } from "./upload-model-dialog"
import { Disease } from "@/types/medical"
import { Button } from "@/components/ui/button"
import { ExternalLink } from 'lucide-react'
interface SettingsTabProps {
  diseases: Disease[]
  onAddDisease: (disease: Omit<Disease, "id" | "is_active" | "created_at">) => Promise<void>
  onModelUploaded: () => void
}

export function SettingsTab({ diseases, onAddDisease, onModelUploaded }: SettingsTabProps) {
  return (
    <div className="space-y-6">
    <div className="flex justify-between items-center">
      <h2 className="text-2xl font-bold tracking-tight">Settings</h2>
      <div className="flex gap-2">
        <Button
          variant="outline"
          onClick={() => window.open('https://kaggle.com', '_blank')}
        >
          <ExternalLink className="mr-2 h-4 w-4" />
          Visit Kaggle
        </Button>
        <AddDiseaseDialog onDiseaseAdded={onAddDisease} />
      </div>
    </div>

      <Card>
        <CardHeader>
          <CardTitle>Available Diseases</CardTitle>
          <CardDescription>
            List of diseases available for analysis. Upload models to activate diseases.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[300px] pr-4">
            <div className="space-y-4">
              {diseases.map((disease) => (
                <Card key={disease.id}>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0">
                    <div>
                      <CardTitle className="text-base">
                        {disease.name}
                        <Badge 
                          variant={disease.is_active ? "default" : "secondary"}
                          className="ml-2"
                        >
                          {disease.is_active ? "Active" : "Inactive"}
                        </Badge>
                      </CardTitle>
                      <CardDescription>{disease.description}</CardDescription>
                    </div>
                    {!disease.is_active && (
                      <UploadModelDialog 
                        disease={disease}
                        onModelUploaded={onModelUploaded}
                      />
                    )}
                  </CardHeader>
                </Card>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}