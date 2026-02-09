'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { FormField, FormFieldWrapper } from './ui/form-field';
import { SubmitButton } from './ui/submit-button';
import { FormTextarea } from './ui/form-textarea';
import { CategorySelectField, ModelSelectField } from './ui/select-field';
import { FileUpload } from './manual/file-upload';
import { FormOptions } from './manual/form-options';
import { useManualGeneration } from '@/lib/hooks/use-manual-generation';
import { useTabs } from '@/lib/hooks/use-tabs';
import { useFileState } from '@/lib/hooks/use-file-state';
import { manualDescriptionSchema } from '@/lib/utils/validation';
import { showErrorToast } from '@/lib/utils/error-handler';
import { GENERATOR_TABS, MESSAGES, FILE_UPLOAD } from '@/lib/constants';
import type { Category } from '@/lib/types/api';

const manualSchema = z.object({
  problem_description: manualDescriptionSchema,
  category: z.string().optional(),
  model: z.string().optional(),
  include_safety: z.boolean().default(true),
  include_tools: z.boolean().default(true),
  include_materials: z.boolean().default(true),
});

type ManualFormData = z.infer<typeof manualSchema>;

export const ManualGenerator = (): JSX.Element => {
  const { files: selectedFiles, setFiles: setSelectedFiles, clearFiles: clearSelectedFiles } = useFileState();
  const { activeTab, setActiveTab } = useTabs({ defaultValue: GENERATOR_TABS.TEXT });
  
  const { register, handleSubmit, formState: { errors }, reset, watch, setValue } = useForm<ManualFormData>({
    resolver: zodResolver(manualSchema),
    defaultValues: {
      include_safety: true,
      include_tools: true,
      include_materials: true,
    },
  });

  const {
    generateFromText,
    generateFromImage,
    generateFromMultipleImages,
    generateCombined,
    isLoading,
  } = useManualGeneration(() => {
    reset();
    clearSelectedFiles();
  });

  const onSubmitText = async (data: ManualFormData): Promise<void> => {
    await generateFromText({
      problemDescription: data.problem_description,
      category: data.category as Category | undefined,
      model: data.model,
      includeSafety: data.include_safety,
      includeTools: data.include_tools,
      includeMaterials: data.include_materials,
    });
  };

  const onSubmitImage = async (data: ManualFormData): Promise<void> => {
    if (selectedFiles.length === 0) {
      showErrorToast(new Error(MESSAGES.MANUAL.NO_IMAGES));
      return;
    }

    const options = {
      problemDescription: data.problem_description || undefined,
      category: data.category as Category | undefined,
      model: data.model,
      includeSafety: data.include_safety,
      includeTools: data.include_tools,
      includeMaterials: data.include_materials,
    };

    if (selectedFiles.length === 1) {
      await generateFromImage(selectedFiles[0], options);
    } else {
      await generateFromMultipleImages(selectedFiles, options);
    }
  };

  const onSubmitCombined = async (data: ManualFormData): Promise<void> => {
    if (!data.problem_description) {
      showErrorToast(new Error(MESSAGES.MANUAL.DESCRIPTION_REQUIRED));
      return;
    }

    await generateCombined(
      data.problem_description,
      selectedFiles[0],
      {
        category: data.category as Category | undefined,
        model: data.model,
        includeSafety: data.include_safety,
        includeTools: data.include_tools,
        includeMaterials: data.include_materials,
      }
    );
  };

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>Generar Manual</CardTitle>
        <CardDescription>
          Crea un manual paso a paso tipo LEGO para resolver problemas del hogar
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value={GENERATOR_TABS.TEXT}>Texto</TabsTrigger>
            <TabsTrigger value={GENERATOR_TABS.IMAGE}>Imagen</TabsTrigger>
            <TabsTrigger value={GENERATOR_TABS.COMBINED}>Combinado</TabsTrigger>
          </TabsList>

          <TabsContent value={GENERATOR_TABS.TEXT} className="space-y-4">
            <form onSubmit={handleSubmit(onSubmitText)} className="space-y-4">
              <FormTextarea
                id="problem_description"
                label="Descripción del Problema"
                {...register('problem_description')}
                placeholder="Describe el problema que necesitas resolver..."
                error={errors.problem_description?.message}
                rows={4}
              />

              <FormFieldWrapper>
                <CategorySelectField
                  label="Categoría"
                  htmlFor="category"
                  value={watch('category') as Category | undefined}
                  onValueChange={(value) => setValue('category', value)}
                />
                <ModelSelectField
                  label="Modelo de IA"
                  htmlFor="model"
                  value={watch('model')}
                  onValueChange={(value) => setValue('model', value)}
                />
              </FormFieldWrapper>

              <FormOptions
                registerSafety={register('include_safety')}
                registerTools={register('include_tools')}
                registerMaterials={register('include_materials')}
              />

              <SubmitButton isLoading={isLoading} loadingText="Generando..." className="w-full">
                Generar Manual
              </SubmitButton>
            </form>
          </TabsContent>

          <TabsContent value={GENERATOR_TABS.IMAGE} className="space-y-4">
            <form onSubmit={handleSubmit(onSubmitImage)} className="space-y-4">
              <FormField label="Imágenes del Problema (máximo 5)" htmlFor="image_files">
                <FileUpload
                  files={selectedFiles}
                  onFilesChange={setSelectedFiles}
                  maxFiles={FILE_UPLOAD.MAX_IMAGES}
                  label="Haz clic o arrastra imágenes aquí"
                  multiple
                />
              </FormField>

              <FormTextarea
                id="image_description"
                label="Descripción Adicional (opcional)"
                {...register('problem_description')}
                placeholder="Describe el problema adicionalmente..."
                rows={3}
              />

              <FormFieldWrapper>
                <CategorySelectField
                  label="Categoría"
                  htmlFor="image_category"
                  value={watch('category') as Category | undefined}
                  onValueChange={(value) => setValue('category', value)}
                  placeholder="Se detectará automáticamente"
                />
                <ModelSelectField
                  label="Modelo de IA"
                  htmlFor="image_model"
                  value={watch('model')}
                  onValueChange={(value) => setValue('model', value)}
                />
              </FormFieldWrapper>

              <SubmitButton
                isLoading={isLoading}
                loadingText="Generando..."
                disabled={selectedFiles.length === 0}
                className="w-full"
              >
                Generar Manual
              </SubmitButton>
            </form>
          </TabsContent>

          <TabsContent value={GENERATOR_TABS.COMBINED} className="space-y-4">
            <form onSubmit={handleSubmit(onSubmitCombined)} className="space-y-4">
              <FormTextarea
                id="combined_description"
                label="Descripción del Problema"
                {...register('problem_description')}
                placeholder="Describe el problema que necesitas resolver..."
                error={errors.problem_description?.message}
                required
                rows={4}
              />

              <FormField label="Imagen (opcional)" htmlFor="combined_image">
                <FileUpload
                  files={selectedFiles}
                  onFilesChange={setSelectedFiles}
                  maxFiles={1}
                  label="Haz clic o arrastra una imagen aquí"
                  multiple={false}
                />
              </FormField>

              <FormFieldWrapper>
                <CategorySelectField
                  label="Categoría"
                  htmlFor="combined_category"
                  value={(watch('category') || 'general') as Category}
                  onValueChange={(value) => setValue('category', value || 'general')}
                />
                <ModelSelectField
                  label="Modelo de IA"
                  htmlFor="combined_model"
                  value={watch('model')}
                  onValueChange={(value) => setValue('model', value)}
                />
              </FormFieldWrapper>

              <SubmitButton isLoading={isLoading} loadingText="Generando..." className="w-full">
                Generar Manual
              </SubmitButton>
            </form>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

