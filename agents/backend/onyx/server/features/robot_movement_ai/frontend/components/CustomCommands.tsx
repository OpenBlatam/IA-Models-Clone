'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion, AnimatePresence } from 'framer-motion';
import { useRobotStore } from '@/lib/store/robotStore';
import { useLocalStorageState } from '@/lib/hooks/useLocalStorageState';
import { handleFormSubmit } from '@/lib/utils/form-helpers';
import { downloadFile, readFileAsText } from '@/lib/utils/file';
import { Command, Plus, Trash2, Play, Save, Load } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import LazyLoad from '@/components/ui/LazyLoad';

const commandSchema = z.object({
  name: z.string().min(1, 'El nombre es requerido').max(50, 'El nombre es demasiado largo'),
  command: z.string().min(1, 'El comando es requerido').max(200, 'El comando es demasiado largo'),
  description: z.string().max(200, 'La descripción es demasiado larga').optional(),
});

type CommandFormData = z.infer<typeof commandSchema>;

interface CustomCommand {
  id: string;
  name: string;
  command: string;
  description?: string;
}

export default function CustomCommands() {
  const { sendChatMessage } = useRobotStore();
  const [commands, setCommands] = useLocalStorageState<CustomCommand[]>(
    'custom-commands',
    []
  );

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
  } = useForm<CommandFormData>({
    resolver: zodResolver(commandSchema),
    defaultValues: {
      name: '',
      command: '',
      description: '',
    },
  });

  const onSubmit = async (data: CommandFormData) => {
    await handleFormSubmit(
      async () => {
        const command: CustomCommand = {
          id: Date.now().toString(),
          name: data.name,
          command: data.command,
          description: data.description,
        };

        setCommands([...commands, command]);
        reset();
        return command;
      },
      {
        successMessage: 'Comando agregado',
        showSuccessToast: true,
      }
    );
  };

  const handleDelete = (id: string) => {
    setCommands(commands.filter((c) => c.id !== id));
    toast.success('Comando eliminado');
  };

  const handleExecute = (command: string) => {
    sendChatMessage(command);
    toast.info('Comando enviado');
  };

  const handleExport = () => {
    try {
      const dataStr = JSON.stringify(commands, null, 2);
      const filename = `custom-commands-${Date.now()}.json`;
      downloadFile(dataStr, filename, 'application/json');
      toast.success('Comandos exportados');
    } catch (error) {
      toast.error('Error al exportar comandos');
    }
  };

  const handleImport = async () => {
    try {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'application/json';
      
      const file = await new Promise<File>((resolve, reject) => {
        input.onchange = (e) => {
          const file = (e.target as HTMLInputElement).files?.[0];
          if (file) {
            resolve(file);
          } else {
            reject(new Error('No file selected'));
          }
        };
        input.click();
      });

      const text = await readFileAsText(file);
      const imported = JSON.parse(text);
      
      if (Array.isArray(imported)) {
        setCommands(imported);
        toast.success('Comandos importados');
      } else {
        toast.error('Formato de archivo inválido');
      }
    } catch (error) {
      toast.error('Error al importar comandos');
    }
  };

  return (
    <div className="space-y-tesla-lg">
      {/* Header */}
      <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-tesla-md mb-tesla-lg">
          <div className="flex items-center gap-tesla-sm">
            <Command className="w-5 h-5 text-tesla-blue" />
            <h3 className="text-lg font-semibold text-tesla-black">Comandos Personalizados</h3>
          </div>
          <div className="flex gap-tesla-sm">
            <button
              onClick={handleExport}
              className="flex items-center gap-tesla-sm px-tesla-md py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black rounded-md transition-all font-medium text-sm min-h-[44px]"
              aria-label="Exportar comandos"
            >
              <Save className="w-4 h-4" />
              Exportar
            </button>
            <button
              onClick={handleImport}
              className="flex items-center gap-tesla-sm px-tesla-md py-tesla-sm bg-tesla-blue hover:bg-opacity-90 text-white rounded-md transition-all font-medium text-sm min-h-[44px]"
              aria-label="Importar comandos"
            >
              <Load className="w-4 h-4" />
              Importar
            </button>
          </div>
        </div>

        {/* Add New Command */}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-tesla-md">
          <Input
            label="Nombre"
            {...register('name')}
            placeholder="Ej: Mover a posición segura"
            error={errors.name?.message}
            aria-label="Nombre del comando"
          />
          <Input
            label="Comando"
            {...register('command')}
            placeholder="Ej: move to (0.5, 0.3, 0.2)"
            error={errors.command?.message}
            className="font-mono text-sm"
            aria-label="Comando a ejecutar"
          />
          <Input
            label="Descripción (opcional)"
            {...register('description')}
            placeholder="Descripción del comando"
            error={errors.description?.message}
            aria-label="Descripción del comando"
          />
          <Button
            type="submit"
            isLoading={isSubmitting}
            className="w-full"
            aria-label="Agregar comando personalizado"
          >
            <Plus className="w-4 h-4" />
            Agregar Comando
          </Button>
        </form>
      </div>

      {/* Commands List */}
      <div className="space-y-tesla-sm">
        <AnimatePresence mode="wait">
          {commands.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white rounded-lg p-tesla-xl border border-gray-200 shadow-sm text-center text-tesla-gray-dark"
            >
              No hay comandos personalizados. Agrega uno para comenzar.
            </motion.div>
          ) : (
            commands.map((cmd, index) => (
              <LazyLoad key={cmd.id} threshold={0.1}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm hover:shadow-tesla-md transition-all"
                >
                  <div className="flex items-start justify-between gap-tesla-md">
                    <div className="flex-1">
                      <h4 className="font-semibold text-tesla-black mb-tesla-sm">{cmd.name}</h4>
                      <p className="text-sm text-tesla-gray-dark font-mono mb-tesla-sm bg-gray-50 p-tesla-sm rounded border border-gray-200">{cmd.command}</p>
                      {cmd.description && (
                        <p className="text-xs text-tesla-gray-dark">{cmd.description}</p>
                      )}
                    </div>
                    <div className="flex gap-tesla-sm">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => handleExecute(cmd.command)}
                        className="p-tesla-sm bg-green-600 hover:bg-green-700 text-white rounded-md transition-all min-h-[44px] min-w-[44px] flex items-center justify-center"
                        title="Ejecutar comando"
                        aria-label={`Ejecutar comando ${cmd.name}`}
                      >
                        <Play className="w-4 h-4" />
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => handleDelete(cmd.id)}
                        className="p-tesla-sm bg-red-600 hover:bg-red-700 text-white rounded-md transition-all min-h-[44px] min-w-[44px] flex items-center justify-center"
                        title="Eliminar comando"
                        aria-label={`Eliminar comando ${cmd.name}`}
                      >
                        <Trash2 className="w-4 h-4" />
                      </motion.button>
                    </div>
                  </div>
                </motion.div>
              </LazyLoad>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

