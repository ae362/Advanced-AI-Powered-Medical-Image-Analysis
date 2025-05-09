'use client'

import { useState } from 'react'
import React from 'react'
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Upload, Trash2 } from 'lucide-react'
import { toast } from "@/components/ui/use-toast"
import { startModelTraining, uploadTrainingImages } from "@/lib/api"

interface ClassConfig {
  key: string;
  name: string;
  uploadedImages: number;
}

export function ModelTraining() {
  const [diseaseName, setDiseaseName] = useState('')
  const [classes, setClasses] = useState<ClassConfig[]>([
    { key: 'positive', name: 'Positive Cases', uploadedImages: 0 },
    { key: 'negative', name: 'Negative Cases', uploadedImages: 0 }
  ])
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [isTraining, setIsTraining] = useState(false)

  const handleAddClass = () => {
    setClasses([...classes, { key: '', name: '', uploadedImages: 0 }])
  }

  const handleDeleteClass = (index: number) => {
    if (index < 2) {
      toast({
        variant: "destructive",
        title: "Cannot Delete",
        description: "Positive and negative classes are required.",
      })
      return
    }
    setClasses(classes.filter((_, i) => i !== index))
  }

  const handleClassChange = (index: number, field: 'key' | 'name', value: string) => {
    const newClasses = [...classes]
    newClasses[index][field] = value
    setClasses(newClasses)
  }

  const handleImageUpload = async (index: number, files: FileList | null) => {
    if (!files) return

    if (!diseaseName.trim()) {
      toast({
        variant: "destructive",
        title: "Disease Name Required",
        description: "Please enter a disease name before uploading images.",
      })
      return
    }

    try {
      const formData = new FormData()
      formData.append('class', classes[index].key)
      formData.append('name', classes[index].name)
      formData.append('disease_name', diseaseName)
      Array.from(files).forEach((file) => {
        formData.append('images', file)
      })

      await uploadTrainingImages(formData)
      
      // Update the uploaded images count for this class
      const newClasses = [...classes]
      newClasses[index].uploadedImages += files.length
      setClasses(newClasses)

      toast({
        title: "Images Uploaded",
        description: `Successfully uploaded ${files.length} images for ${classes[index].name}.`,
      })
    } catch (error) {
      console.error('Error uploading images:', error)
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: "There was an error uploading the images. Please try again.",
      })
    }
  }

  const handleStartTraining = async () => {
    if (!diseaseName.trim()) {
      toast({
        variant: "destructive",
        title: "Disease Name Required",
        description: "Please enter a disease name before starting training.",
      })
      return
    }

    // Check if all classes have uploaded images
    const classWithoutImages = classes.find(c => c.uploadedImages === 0)
    if (classWithoutImages) {
      toast({
        variant: "destructive",
        title: "Missing Training Data",
        description: `Please upload images for ${classWithoutImages.name} before training.`,
      })
      return
    }

    setIsTraining(true)
    setTrainingProgress(0)

    try {
      const eventSource = new EventSource('http://localhost:8000/api/training/start/stream/')
    
      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.epoch) {
          // Calculate progress as current epoch / total epochs * 100
          const progress = (data.epoch / data.totalEpochs) * 100
          setTrainingProgress(progress)
        }
      }

      eventSource.onerror = () => {
        eventSource.close()
      }

      const result = await startModelTraining({
        diseaseName,
        classes
      })

      eventSource.close()
      setTrainingProgress(100)
      toast({
        title: "Training Complete",
        description: "Model training has been completed successfully.",
      })
    } catch (error) {
      console.error('Error during model training:', error)
      toast({
        variant: "destructive",
        title: "Training Failed",
        description: "There was an error during model training. Please try again.",
      })
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardContent className="p-6">
        <h2 className="text-2xl font-bold mb-6">Model Training</h2>
        
        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">Disease Name</label>
            <Input
              value={diseaseName}
              onChange={(e) => setDiseaseName(e.target.value)}
              placeholder="Enter disease name"
              className="max-w-md"
            />
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold">Classes</h3>
              <Button 
                variant="secondary" 
                onClick={handleAddClass} 
                disabled={classes.length >= 5}
              >
                + Add Class
              </Button>
            </div>

            <div className="grid grid-cols-[1fr,1fr,auto,auto] gap-x-4 gap-y-4">
              <div className="font-medium text-sm">Class Key</div>
              <div className="font-medium text-sm">Class Name</div>
              <div className="font-medium text-sm text-center">Images</div>
              <div></div>
              
              {classes.map((classConfig, index) => (
                <React.Fragment key={index}>
                  <div>
                    <Input
                      value={classConfig.key}
                      onChange={(e) => handleClassChange(index, 'key', e.target.value)}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <Input
                      value={classConfig.name}
                      onChange={(e) => handleClassChange(index, 'name', e.target.value)}
                      className="w-full"
                    />
                  </div>
                  <div className="flex items-center justify-center">
                    <span className="text-sm text-muted-foreground">
                      {classConfig.uploadedImages} uploaded
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <label htmlFor={`file-upload-${index}`} className="cursor-pointer">
                      <div className="flex items-center justify-center w-10 h-10 rounded-md bg-primary text-primary-foreground">
                        <Upload className="h-5 w-5" />
                      </div>
                      <input
                        id={`file-upload-${index}`}
                        type="file"
                        multiple
                        accept=".jpg,.jpeg,.png"
                        className="hidden"
                        onChange={(e) => handleImageUpload(index, e.target.files)}
                      />
                    </label>
                    <Button
                      variant="destructive"
                      size="icon"
                      onClick={() => handleDeleteClass(index)}
                      className="w-10 h-10"
                    >
                      <Trash2 className="h-5 w-5" />
                    </Button>
                  </div>
                </React.Fragment>
              ))}
            </div>
          </div>

          <Button 
            onClick={handleStartTraining} 
            disabled={isTraining}
            className="w-auto"
          >
            Start Training
          </Button>

          {trainingProgress > 0 && (
            <div className="space-y-2">
              <Progress value={trainingProgress} className="w-full" />
              <p className="text-sm text-muted-foreground text-center">
                Training Progress: {Math.round(trainingProgress)}% 
                {isTraining && " - Training in progress..."}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
