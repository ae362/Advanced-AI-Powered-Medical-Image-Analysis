'use client'

import { useTranslations } from 'next-intl'
import { useRouter } from 'next/navigation'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export function LanguageSelector() {
  const t = useTranslations('settings')
  const tLanguages = useTranslations('languages')
  const router = useRouter()

  const changeLanguage = (newLocale: string) => {
    router.push(`/${newLocale}${window.location.pathname.substring(3)}`)
  }

  return (
    <div className="flex flex-col space-y-2">
      <label htmlFor="language-select" className="text-sm font-medium">
        {t('languageSelector')}
      </label>
      <Select onValueChange={changeLanguage} defaultValue={window.location.pathname.substring(1, 3)}>
        <SelectTrigger id="language-select">
          <SelectValue placeholder={t('selectLanguage')} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="en">{tLanguages('en')}</SelectItem>
          <SelectItem value="fr">{tLanguages('fr')}</SelectItem>
          <SelectItem value="es">{tLanguages('es')}</SelectItem>
          <SelectItem value="ar">{tLanguages('ar')}</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}

