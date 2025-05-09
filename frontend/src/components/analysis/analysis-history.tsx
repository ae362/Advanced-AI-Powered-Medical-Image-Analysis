'use client'

import { useState, useRef, useEffect } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Slider } from "@/components/ui/slider"
import { toast } from "@/components/ui/use-toast"
import { Patient, Analysis } from "@/types/medical"
import { formatDate } from "@/lib/utils"
import { saveAnalysisNote, deleteAnalysisNotes, getAIExplanationAndStaging, updateNotificationEmail ,sendAnalysisToEmail} from "@/lib/api"
import { PrinterIcon, Share2Icon, Trash2Icon, ZoomInIcon, ZoomOutIcon, ContrastIcon, MailIcon, Loader2,Mail } from 'lucide-react'
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import { AnalysisTrends } from "./analysis-trends"
interface AnalysisHistoryProps {
  patient: Patient
  analyses: Analysis[]
  onDeleteAnalysis: (analysisId: string) => void
}

interface Note {
  x: number
  y: number
  text: string
}

export function AnalysisHistory({ patient, analyses, onDeleteAnalysis }: AnalysisHistoryProps) {
  const [selectedAnalysis, setSelectedAnalysis] = useState<Analysis | null>(null)
  const [zoom, setZoom] = useState(1)
  const [contrast, setContrast] = useState(100)
  const [notes, setNotes] = useState<Note[]>([])
  const [isAddingNote, setIsAddingNote] = useState(false)
  const [notesModified, setNotesModified] = useState(false)
  const imageRef = useRef<HTMLDivElement>(null)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [explanation, setExplanation] = useState<string | null>(null)
  const [isGeneratingExplanation, setIsGeneratingExplanation] = useState(false)
  const [notificationEmail, setNotificationEmail] = useState('')
  const [isUpdatingEmail, setIsUpdatingEmail] = useState(false)
  const [isSendingEmail, setIsSendingEmail] = useState(false)
  const [customEmail, setCustomEmail] = useState("")
  const [isSendingCustomEmail, setIsSendingCustomEmail] = useState(false)
  useEffect(() => {
    if (selectedAnalysis) {
      setZoom(1)
      setContrast(100)
      setPan({ x: 0, y: 0 })
      setExplanation(null)
      setNotificationEmail(selectedAnalysis.notificationEmail || '')
      try {
        const parsedNotes = selectedAnalysis.notes ? 
          (typeof selectedAnalysis.notes === 'string' ? 
            JSON.parse(selectedAnalysis.notes) : 
            selectedAnalysis.notes) : 
          [];
        setNotes(Array.isArray(parsedNotes) ? parsedNotes : []);
        setNotesModified(false)
      } catch (error) {
        console.error('Error parsing notes:', error);
        setNotes([]);
      }
    }

    return () => {
      if (selectedAnalysis && notes.length > 0) {
        deleteAnalysisNotes(selectedAnalysis.id).catch(console.error);
      }
    }
  }, [selectedAnalysis])

  useEffect(() => {
    return () => {
      if (selectedAnalysis && notes.length > 0) {
        deleteAnalysisNotes(selectedAnalysis.id).catch(console.error);
      }
    }
  }, [])

  const handleSelectAnalysis = async (analysis: Analysis) => {
    if (selectedAnalysis && notes.length > 0) {
      try {
        await deleteAnalysisNotes(selectedAnalysis.id);
      } catch (error) {
        console.error('Error deleting notes:', error);
      }
    }
    setSelectedAnalysis(analysis);
  }

  const handlePrint = () => {
    if (!selectedAnalysis) return
    window.print()
  }

  const handleShare = async () => {
    if (!selectedAnalysis) return
    try {
      await navigator.share({
        title: `Medical Analysis Result - ${patient.name}`,
        text: `Analysis Result: ${selectedAnalysis.prediction} (Confidence: ${(selectedAnalysis.confidence * 100).toFixed(2)}%)`,
      })
    } catch (error) {
      console.error('Error sharing:', error)
    }
  }

  const handleDelete = () => {
    if (!selectedAnalysis) return
    onDeleteAnalysis(selectedAnalysis.id)
    setSelectedAnalysis(null)
  }

  const handleZoomIn = () => {
    setZoom(prevZoom => Math.min(prevZoom + 0.1, 3))
  }

  const handleZoomOut = () => {
    setZoom(prevZoom => Math.max(prevZoom - 0.1, 0.5))
  }

  const handleContrastChange = (value: number[]) => {
    setContrast(value[0])
  }

  const handleImageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageRef.current || !selectedAnalysis) return
    const rect = imageRef.current.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    
    const menu = document.createElement('div')
    menu.className = 'absolute bg-popover text-popover-foreground border rounded-md shadow-md p-1 z-50'
    menu.style.left = `${e.clientX}px`
    menu.style.top = `${e.clientY}px`
    
    const addNoteButton = document.createElement('button')
    addNoteButton.className = 'w-full text-left px-2 py-1 text-sm hover:bg-muted rounded-sm'
    addNoteButton.textContent = 'Add note'
    addNoteButton.onclick = () => {
      setIsAddingNote(true)
      setNotes(prevNotes => [...prevNotes, { x, y, text: '' }])
      document.body.removeChild(menu)
    }
    
    menu.appendChild(addNoteButton)
    document.body.appendChild(menu)
    
    const handleClickOutside = () => {
      if (document.body.contains(menu)) {
        document.body.removeChild(menu)
      }
      document.removeEventListener('click', handleClickOutside)
    }
    
    setTimeout(() => {
      document.addEventListener('click', handleClickOutside)
    }, 0)
  }

  const handleNoteTextChange = (index: number, text: string) => {
    setNotes(prevNotes => prevNotes.map((note, i) => i === index ? { ...note, text } : note))
    setNotesModified(true)
  }

  const handleNoteSave = async () => {
    if (!selectedAnalysis) return
    try {
      const notesString = JSON.stringify(notes)
      await saveAnalysisNote(selectedAnalysis.id, notesString)
      setNotesModified(false)
      toast({
        title: "Notes saved",
        description: "The analysis notes have been successfully saved.",
      })
      setIsAddingNote(false)
    } catch (error) {
      console.error('Error saving notes:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to save the analysis notes. Please try again.",
      })
    }
  }

  const handleDeleteNotes = async () => {
    if (!selectedAnalysis) return
    try {
      await deleteAnalysisNotes(selectedAnalysis.id)
      setNotes([])
      setNotesModified(false)
      toast({
        title: "Notes deleted",
        description: "All notes for this analysis have been deleted.",
      })
    } catch (error) {
      console.error('Error deleting notes:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete the analysis notes. Please try again.",
      })
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true)
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging) return
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleGenerateExplanationAndStaging = async () => {
    if (!selectedAnalysis) return
    setIsGeneratingExplanation(true)
    try {
      const result = await getAIExplanationAndStaging(selectedAnalysis.id)
      setExplanation(result.analysis)
    } catch (error) {
      console.error('Error generating explanation and staging:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to generate AI explanation and staging. Please try again.",
      })
    } finally {
      setIsGeneratingExplanation(false)
    }
  }
  
  const handleSendToEmail = async () => {
    if (!selectedAnalysis || !customEmail) return;
    
    setIsSendingCustomEmail(true);
    try {
      await sendAnalysisToEmail(selectedAnalysis.id, customEmail);
      toast({
        title: "Email Sent",
        description: "Analysis results have been sent to the specified email.",
      });
      setCustomEmail(""); // Clear the input after successful send
    } catch (error) {
      console.error('Error sending email:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to send email. Please try again.",
      });
    } finally {
      setIsSendingCustomEmail(false);
    }
  };

  const handleUpdateEmail = async () => {
    if (!selectedAnalysis) return
    setIsUpdatingEmail(true)
    try {
      await updateNotificationEmail(selectedAnalysis.id, notificationEmail)
      toast({
        title: "Email Updated",
        description: "Notification email updated successfully",
      })
    } catch (error) {
      console.error('Error updating notification email:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to update notification email. Please try again.",
      })
    } finally {
      setIsUpdatingEmail(false)
    }
  }

  return (

  <div className="space-y-6">
    <AnalysisTrends analyses={analyses} />
    <div className="grid lg:grid-cols-[350px,1fr] gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Analysis History</CardTitle>
          <CardDescription>
            Patient: {patient.name}
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-[600px] p-4">
            <div className="space-y-4">
              {analyses.map((analysis) => (
                <Card
                  key={analysis.id}
                  className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                    selectedAnalysis?.id === analysis.id ? 'bg-muted' : ''
                  }`}
                  onClick={() => handleSelectAnalysis(analysis)}
                >
                  <CardHeader className="p-4">
                    <div className="flex items-center justify-between">
                      <Badge variant={analysis.type === 'brain_tumor' ? 'default' : 'secondary'}>
                        {analysis.type === 'brain_tumor' ? 'Brain Tumor' : 'Cancer'}
                      </Badge>
                      <Badge variant={analysis.prediction === 'positive' ? 'destructive' : 'success'}>
                        {analysis.prediction}
                      </Badge>
                    </div>
                    <CardDescription>
                      {analysis.createdAt ? formatDate(analysis.createdAt) : formatDate(new Date())}
                    </CardDescription>
                  </CardHeader>
                </Card>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      <Card>
        {selectedAnalysis ? (
          <>
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <div>
                <CardTitle>Analysis Result</CardTitle>
                <CardDescription>
                  {formatDate(selectedAnalysis.date)}
                </CardDescription>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="icon" onClick={handleShare}>
                  <Share2Icon className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon" onClick={handlePrint}>
                  <PrinterIcon className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon" onClick={handleDelete}>
                  <Trash2Icon className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 rounded-lg border p-4">
                <div className="flex justify-between">
                  <span className="font-medium">Type:</span>
                  <Badge variant={selectedAnalysis.type === 'brain_tumor' ? 'default' : 'secondary'}>
                    {selectedAnalysis.type === 'brain_tumor' ? 'Brain Tumor' : 'Cancer'}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Diagnosis:</span>
                  <Badge 
                    variant={selectedAnalysis.prediction === 'positive' ? 'destructive' : 'outline'}
                    className={cn(
                      "capitalize font-medium",
                      selectedAnalysis.prediction === 'positive' 
                        ? "text-destructive-foreground" 
                        : "text-green-600 dark:text-green-500 border-green-600 dark:border-green-500"
                    )}
                  >
                    {selectedAnalysis.prediction === 'positive' ? 'Abnormal' : 'Normal'}
                  </Badge>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="font-medium">Confidence:</span>
                    <span className="text-sm text-muted-foreground">
                      {(selectedAnalysis.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  <Progress 
                    value={selectedAnalysis.confidence * 100} 
                    className="h-2"
                    indicatorClassName={cn(
                      selectedAnalysis.prediction === 'positive' 
                        ? "bg-destructive" 
                        : "bg-green-600 dark:bg-green-500"
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="font-medium">Model Accuracy:</span>
                    <span className="text-sm text-muted-foreground">
                      {(selectedAnalysis.modelAccuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                  <Progress 
                    value={selectedAnalysis.modelAccuracy * 100} 
                    className="h-2"
                    indicatorClassName="bg-cyan-500 dark:bg-cyan-400"
                  />
                </div>
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                  <Input
                    type="email"
                    placeholder="Enter email to send results"
                    value={customEmail}
                    onChange={(e) => setCustomEmail(e.target.value)}
                    className="w-64"
                  />
                    <Button onClick={handleSendToEmail} disabled={!customEmail || isSendingCustomEmail} >
                    {isSendingCustomEmail ? (
                      <Loader2 className="animate-spin h-4 w-4" />
                    ) : (
                      <Mail className="h-4 w-4" />
                    )}
                      Send results
                    </Button>
                  </div>
                  <Button 
                    onClick={handleGenerateExplanationAndStaging} 
                    disabled={isGeneratingExplanation}
                  >
                    {isGeneratingExplanation ? 'Generating...' : 'Generate Explanation'}
                  </Button>
                  {explanation && (
                    <Card>
                      <CardHeader>
                        <CardTitle>AI-Generated Explanation</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <pre className="whitespace-pre-wrap bg-muted p-4 rounded-lg text-sm">
                          {explanation}
                        </pre>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="font-medium">Visualization</h4>
                <div 
                  className="overflow-hidden rounded-lg border cursor-move relative aspect-square"
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                >
                  <div
                    ref={imageRef}
                    className="relative w-full h-full"
                    style={{
                      width: '100%',
                      height: '100%',
                      overflow: 'hidden',
                    }}
                    onClick={handleImageClick}
                  >
                    <img
                      src={`data:image/png;base64,${selectedAnalysis.visualization}`}
                      alt="Analysis Visualization"
                      className="w-full h-full object-contain"
                      style={{
                        transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                        filter: `contrast(${contrast}%)`,
                        transformOrigin: 'center',
                        transition: 'transform 0.1s ease-out',
                      }}
                    />
                    {notes.map((note, index) => (
                      <div
                        key={index}
                        style={{
                          position: 'absolute',
                          left: `${note.x * 100}%`,
                          top: `${note.y * 100}%`,
                          transform: 'translate(-50%, -50%)',
                        }}
                      >
                        <div className="relative group">
                          <Badge variant="secondary" className="cursor-pointer">
                            {index + 1}
                          </Badge>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="absolute -top-2 -right-2 h-4 w-4 p-0 rounded-full bg-destructive text-destructive-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={(e) => {
                              e.stopPropagation()
                              setNotes(prevNotes => prevNotes.filter((_, i) => i !== index))
                              handleNoteSave()
                            }}
                          >
                            <Trash2Icon className="h-3 w-3" />
                          </Button>
                          {isAddingNote && index === notes.length - 1 && (
                            <Input
                              value={note.text}
                              onChange={(e) => handleNoteTextChange(index, e.target.value)}
                              onBlur={handleNoteSave}
                              className="absolute top-full left-0 mt-1 w-40"
                              placeholder="Add note..."
                            />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="flex justify-between items-center mt-2">
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={handleZoomOut}>
                      <ZoomOutIcon className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="sm" onClick={handleZoomIn}>
                      <ZoomInIcon className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="flex items-center gap-2">
                    <ContrastIcon className="h-4 w-4" />
                    <Slider
                      value={[contrast]}
                      min={50}
                      max={150}
                      step={1}
                      onValueChange={handleContrastChange}
                      className="w-[100px]"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <h4 className="font-medium">Notes</h4>
                  <Button variant="outline" size="sm" onClick={handleDeleteNotes}>
                    Delete All Notes
                  </Button>
                </div>
                <ScrollArea className="h-[200px] w-full rounded-md border p-4">
                  {notes.map((note, index) => (
                    <div key={index} className="mb-2 flex items-center justify-between">
                      <div className="flex items-center">
                        <Badge variant="secondary" className="mr-2">{index + 1}</Badge>
                        {note.text}
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                        onClick={() => {
                          setNotes(prevNotes => prevNotes.filter((_, i) => i !== index))
                          setNotesModified(true)
                        }}
                      >
                        <Trash2Icon className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </ScrollArea>
              </div>
            </CardContent>
          </>
        ) : (
          <CardContent className="min-h-[600px] flex items-center justify-center text-muted-foreground">
            Select an analysis to view details
          </CardContent>
        )}
      </Card>
    </div>
    </div>
  )
}

