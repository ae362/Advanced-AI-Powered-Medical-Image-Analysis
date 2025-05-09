import {getRequestConfig} from 'next-intl/server';
 
export default getRequestConfig(async ({locale}) => {
  // Validate that the incoming `locale` parameter is valid
  const locales = ['en', 'fr', 'es', 'ar'];
  if (!locales.includes(locale as any)) {
    return {
      messages: (await import(`../messages/en.json`)).default,
      locale: 'en'
    };
  }
 
  return {
    messages: (await import(`../messages/${locale}.json`)).default,
    locale: locale
  };
});

