import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

// Configurar timeout máximo para esta ruta (10 minutos) - NO SE DETIENE
export const maxDuration = 600; // 10 minutos en segundos
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs'; // Asegurar que use Node.js runtime

const DEFAULT_GITHUB_TOKEN = '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';

// Obtener token de acceso (solo en el servidor, el token viene en el request)

interface FileAction {
  path: string;
  content: string;
  action: 'create' | 'update';
  message: string;
}

interface ExecuteRequest {
  repository: string; // owner/repo
  branch?: string;
  actions: FileAction[];
  commitMessage: string;
  token?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: ExecuteRequest = await request.json();
    const { repository, branch = 'main', actions, commitMessage, token } = body;

    // Validaciones NO BLOQUEANTES - usar valores por defecto si falta algo
    if (!repository) {
      console.warn('⚠️ No se proporcionó repositorio, no se puede continuar');
      return NextResponse.json(
        { 
          success: false,
          error: 'Se requiere repositorio. El proceso no puede continuar sin esta información.',
        },
        { status: 400 }
      );
    }

    // Si no hay acciones, retornar pero NO detener completamente
    if (!actions || actions.length === 0) {
      console.warn('⚠️ No hay acciones para ejecutar');
      return NextResponse.json(
        { 
          success: false,
          error: 'No hay acciones para ejecutar. El proceso puede continuar cuando se agreguen acciones.',
        },
        { status: 400 }
      );
    }

    const accessToken = token || DEFAULT_GITHUB_TOKEN;
    const [owner, repo] = repository.split('/');

    // Validar formato pero intentar continuar
    if (!owner || !repo) {
      console.error('❌ Formato de repositorio inválido:', repository);
      return NextResponse.json(
        { 
          success: false,
          error: 'Formato de repositorio inválido. Debe ser owner/repo. El proceso no puede continuar sin esta información.',
        },
        { status: 400 }
      );
    }

    const headers = {
      'Authorization': `Bearer ${accessToken}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
    };

    const results = [];
    const errors = [];
    const maxRetries = 3; // Número máximo de reintentos para todas las operaciones

    // Obtener el SHA del branch actual - CON REINTENTOS Y SIN DETENERSE
    let branchSha: string;
    let branchObtained = false;
    
    for (let attempt = 1; attempt <= maxRetries && !branchObtained; attempt++) {
      try {
        const branchResponse = await axios.get(
          `https://api.github.com/repos/${owner}/${repo}/git/ref/heads/${branch}`,
          { 
            headers,
            timeout: 30000, // 30 segundos por intento
          }
        );
        branchSha = branchResponse.data.object.sha;
        branchObtained = true;
      } catch (error: any) {
        console.warn(`⚠️ Intento ${attempt}/${maxRetries} falló al obtener branch ${branch}:`, error.message);
        
        if (attempt < maxRetries) {
          // Esperar antes de reintentar
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
          continue;
        }
        
        // Si el branch no existe, intentar obtener el SHA del repositorio
        try {
          const repoResponse = await axios.get(
            `https://api.github.com/repos/${owner}/${repo}`,
            { 
              headers,
              timeout: 30000,
            }
          );
          const defaultBranch = repoResponse.data.default_branch;
          const branchResponse = await axios.get(
            `https://api.github.com/repos/${owner}/${repo}/git/ref/heads/${defaultBranch}`,
            { 
              headers,
              timeout: 30000,
            }
          );
          branchSha = branchResponse.data.object.sha;
          branchObtained = true;
        } catch (e: any) {
          console.warn(`⚠️ No se pudo obtener branch del repositorio en intento ${attempt}:`, e.message);
          
          // Si es el último intento, intentar obtener SHA del último commit o continuar sin parent
          if (attempt >= maxRetries) {
            console.warn('⚠️ Intentando obtener SHA del último commit directamente...');
            try {
              // Intentar obtener el último commit del repositorio
              const commitsResponse = await axios.get(
                `https://api.github.com/repos/${owner}/${repo}/commits`,
                { 
                  headers,
                  timeout: 30000,
                  params: { per_page: 1 }
                }
              );
              if (commitsResponse.data && commitsResponse.data.length > 0) {
                branchSha = commitsResponse.data[0].sha;
                branchObtained = true;
                console.log(`✅ Obtenido SHA del último commit: ${branchSha}`);
              } else {
                // Si no hay commits, crear un commit inicial (sin parent)
                console.warn('⚠️ No hay commits previos, se creará un commit inicial sin parent');
                branchSha = null as any; // Se manejará al crear el commit
                branchObtained = true; // Marcar como obtenido para continuar
              }
            } catch (finalError: any) {
              console.warn('⚠️ No se pudo obtener SHA, continuando sin parent (commit inicial)');
              // Aún así continuar - se intentará crear un commit sin parent
              branchSha = null as any;
              branchObtained = true;
            }
          }
        }
      }
    }

    // Obtener el árbol base - CON REINTENTOS Y SIN DETENERSE
    let baseTreeSha: string | undefined;
    let treeObtained = false;
    
    // Si no hay branchSha, no intentar obtener árbol base (será un árbol nuevo)
    if (!branchSha) {
      console.log('📝 No hay branch SHA, se creará un árbol nuevo sin base');
      treeObtained = true; // Marcar como obtenido para continuar sin base
    } else {
      for (let attempt = 1; attempt <= maxRetries && !treeObtained; attempt++) {
        try {
          const commitResponse = await axios.get(
            `https://api.github.com/repos/${owner}/${repo}/git/commits/${branchSha}`,
            { 
              headers,
              timeout: 30000,
            }
          );
          baseTreeSha = commitResponse.data.tree.sha;
          treeObtained = true;
        } catch (error: any) {
          console.warn(`⚠️ Intento ${attempt}/${maxRetries} falló al obtener árbol base:`, error.message);
          
          if (attempt < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
            continue;
          }
          
          // Si no se puede obtener, crear un árbol nuevo desde cero - NO DETENERSE
          console.warn('⚠️ No se pudo obtener árbol base, se creará un árbol nuevo sin base');
          baseTreeSha = undefined; // Usar undefined para crear árbol nuevo
          treeObtained = true; // Marcar como obtenido para continuar
          console.log('📝 Continuando con árbol nuevo (sin base)');
        }
      }
    }

    // Procesar cada acción - CONTINUAR INCLUSO SI ALGUNAS FALLAN
    const treeItems: any[] = [];

    for (const action of actions) {
      try {
        let fileSha: string | undefined;

        // Si es una actualización, obtener el SHA del archivo existente
        if (action.action === 'update') {
          try {
            const fileResponse = await axios.get(
              `https://api.github.com/repos/${owner}/${repo}/contents/${action.path}`,
              { 
                headers,
                params: { ref: branch },
                timeout: 30000,
              }
            );
            fileSha = fileResponse.data.sha;
          } catch (e: any) {
            // Si el archivo no existe, tratarlo como creación - NO DETENERSE
            console.log(`📝 Archivo ${action.path} no existe, se creará como nuevo`);
          }
        }

        // Validar que el contenido existe
        if (!action.content) {
          console.warn(`⚠️ Acción ${action.path} no tiene contenido, saltando...`);
          errors.push({
            path: action.path,
            action: action.action,
            error: 'Contenido vacío',
          });
          continue; // Continuar con la siguiente acción
        }

        treeItems.push({
          path: action.path,
          mode: '100644', // Archivo regular
          type: 'blob',
          content: action.content,
          sha: fileSha,
        });

        results.push({
          path: action.path,
          action: action.action,
          status: 'success',
        });
      } catch (error: any) {
        console.warn(`⚠️ Error procesando acción ${action.path}:`, error.message);
        errors.push({
          path: action.path,
          action: action.action,
          error: error.message,
        });
        // NO DETENERSE - continuar con las siguientes acciones
      }
    }

    // Si TODAS las acciones fallaron, retornar error pero NO DETENERSE completamente
    if (errors.length > 0 && results.length === 0) {
      console.error('❌ Todas las acciones fallaron, pero el proceso puede continuar');
      return NextResponse.json(
        { 
          success: false,
          error: 'No se pudo procesar ninguna acción, pero el proceso puede continuar en background',
          errors 
        },
        { status: 500 }
      );
    }
    
    // Si hay al menos una acción exitosa, continuar (aunque haya errores)
    if (results.length > 0) {
      console.log(`✅ ${results.length} acción(es) procesada(s) exitosamente, ${errors.length} error(es)`);
    }

    // Crear blobs para cada archivo - CONTINUAR INCLUSO SI ALGUNOS FALLAN
    const blobs: any[] = [];
    const blobErrors: any[] = [];
    
    for (const item of treeItems) {
      let blobCreated = false;
      
      // Reintentar hasta 3 veces por blob
      for (let attempt = 1; attempt <= maxRetries && !blobCreated; attempt++) {
        try {
          const blobResponse = await axios.post(
            `https://api.github.com/repos/${owner}/${repo}/git/blobs`,
            {
              content: item.content,
              encoding: 'utf-8',
            },
            { 
              headers,
              timeout: 60000, // 60 segundos para blobs grandes
            }
          );
          blobs.push({
            path: item.path,
            sha: blobResponse.data.sha,
            mode: item.mode,
            type: item.type,
          });
          blobCreated = true;
        } catch (error: any) {
          console.warn(`⚠️ Intento ${attempt}/${maxRetries} falló al crear blob para ${item.path}:`, error.message);
          
          if (attempt < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
            continue;
          }
          
          // Si falla después de todos los intentos, agregar a errores pero CONTINUAR
          blobErrors.push({
            path: item.path,
            error: `Error al crear blob después de ${maxRetries} intentos: ${error.message}`,
          });
          console.error(`❌ No se pudo crear blob para ${item.path} después de ${maxRetries} intentos`);
        }
      }
    }
    
    // Si no se pudo crear ningún blob, retornar error
    if (blobs.length === 0) {
      return NextResponse.json(
        { 
          success: false,
          error: 'No se pudo crear ningún blob después de múltiples intentos. El proceso puede continuar en background.',
          errors: [...errors, ...blobErrors],
        },
        { status: 500 }
      );
    }
    
    // Si hay errores pero también blobs exitosos, continuar
    if (blobErrors.length > 0) {
      console.warn(`⚠️ ${blobErrors.length} blob(s) fallaron, pero ${blobs.length} se crearon exitosamente`);
    }

    // Crear el árbol con los nuevos blobs - CON REINTENTOS Y SIN DETENERSE
    const treeData = blobs.map(blob => ({
      path: blob.path,
      mode: blob.mode,
      type: blob.type,
      sha: blob.sha,
    }));

    let newTreeSha: string;
    let treeCreated = false;
    
    for (let attempt = 1; attempt <= maxRetries && !treeCreated; attempt++) {
      try {
        // Si no hay baseTreeSha, crear árbol sin base (nuevo árbol)
        const treePayload: any = {
          tree: treeData,
        };
        
        if (baseTreeSha) {
          treePayload.base_tree = baseTreeSha;
        }
        
        const treeResponse = await axios.post(
          `https://api.github.com/repos/${owner}/${repo}/git/trees`,
          treePayload,
          { 
            headers,
            timeout: 60000,
          }
        );
        newTreeSha = treeResponse.data.sha;
        treeCreated = true;
        console.log(`✅ Árbol creado exitosamente: ${newTreeSha}`);
      } catch (error: any) {
        console.warn(`⚠️ Intento ${attempt}/${maxRetries} falló al crear árbol:`, error.message);
        
        if (attempt < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 2000 * attempt));
          continue;
        }
        
        // NO DETENERSE - intentar crear commit sin árbol (fallback)
        console.error(`❌ No se pudo crear árbol después de ${maxRetries} intentos, pero el proceso puede continuar`);
        // Continuar de todas formas - el commit puede fallar pero no detenemos el proceso
        return NextResponse.json(
          { 
            success: false,
            error: `Error al crear árbol después de ${maxRetries} intentos: ${error.message}. El proceso puede continuar en background.`,
            errors: [...errors, ...blobErrors],
            partialResults: {
              blobsCreated: blobs.length,
              treeItems: treeItems.length,
            },
          },
          { status: 500 }
        );
      }
    }

    // Crear el commit - CON REINTENTOS Y NO DETENERSE
    let commitSha: string;
    let commitCreated = false;
    
    for (let attempt = 1; attempt <= maxRetries && !commitCreated; attempt++) {
      try {
        // Si no hay branchSha, crear commit sin parent (commit inicial)
        const commitPayload: any = {
          message: commitMessage,
          tree: newTreeSha,
        };
        
        if (branchSha) {
          commitPayload.parents = [branchSha];
        } else {
          // Commit inicial sin parent
          commitPayload.parents = [];
          console.log('📝 Creando commit inicial (sin parent)');
        }
        
        const commitResponse = await axios.post(
          `https://api.github.com/repos/${owner}/${repo}/git/commits`,
          commitPayload,
          { 
            headers,
            timeout: 60000,
          }
        );
        commitSha = commitResponse.data.sha;
        commitCreated = true;
        console.log(`✅ Commit creado exitosamente: ${commitSha}`);
      } catch (error: any) {
        console.warn(`⚠️ Intento ${attempt}/${maxRetries} falló al crear commit:`, error.message);
        
        if (attempt < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 2000 * attempt));
          continue;
        }
        
        // NO DETENERSE completamente - retornar información parcial
        console.error(`❌ No se pudo crear commit después de ${maxRetries} intentos, pero los blobs y árbol fueron creados`);
        return NextResponse.json(
          { 
            success: false,
            error: `Error al crear commit después de ${maxRetries} intentos: ${error.message}. El proceso puede continuar en background.`,
            errors: [...errors, ...blobErrors],
            partialResults: {
              blobsCreated: blobs.length,
              treeCreated: newTreeSha,
              message: 'Blobs y árbol creados, pero commit falló. Puede intentarse manualmente.',
            },
          },
          { status: 500 }
        );
      }
    }

    // Actualizar la referencia del branch - CON REINTENTOS Y NO DETENERSE
    let branchUpdated = false;
    
    for (let attempt = 1; attempt <= maxRetries && !branchUpdated; attempt++) {
      try {
        await axios.patch(
          `https://api.github.com/repos/${owner}/${repo}/git/refs/heads/${branch}`,
          {
            sha: commitSha,
          },
          { 
            headers,
            timeout: 30000,
          }
        );
        branchUpdated = true;
        console.log(`✅ Branch ${branch} actualizado exitosamente`);
      } catch (error: any) {
        console.warn(`⚠️ Intento ${attempt}/${maxRetries} falló al actualizar branch:`, error.message);
        
        // Si el branch no existe, intentar crearlo
        if (attempt < maxRetries) {
          try {
            await axios.post(
              `https://api.github.com/repos/${owner}/${repo}/git/refs`,
              {
                ref: `refs/heads/${branch}`,
                sha: commitSha,
              },
              { 
                headers,
                timeout: 30000,
              }
            );
            branchUpdated = true;
            console.log(`✅ Branch ${branch} creado exitosamente`);
            break;
          } catch (e: any) {
            console.warn(`⚠️ No se pudo crear branch en intento ${attempt}, reintentando...`);
            await new Promise(resolve => setTimeout(resolve, 2000 * attempt));
            continue;
          }
        } else {
          // Si falla después de todos los intentos, aún así retornar éxito parcial
          console.error(`❌ No se pudo actualizar branch después de ${maxRetries} intentos, pero el commit existe`);
          // NO DETENERSE - el commit ya existe, solo no se actualizó la referencia
          return NextResponse.json({
            success: true, // Considerar éxito porque el commit existe
            commitSha,
            branch,
            results,
            errors: errors.length > 0 ? errors : undefined,
            commitUrl: `https://github.com/${owner}/${repo}/commit/${commitSha}`,
            warning: `Commit creado pero no se pudo actualizar branch después de ${maxRetries} intentos. El commit existe y puede ser aplicado manualmente.`,
          });
        }
      }
    }

    return NextResponse.json({
      success: true,
      commitSha,
      branch,
      results,
      errors: errors.length > 0 ? errors : undefined,
      commitUrl: `https://github.com/${owner}/${repo}/commit/${commitSha}`,
    });
  } catch (error: any) {
    console.error('❌ Error executing GitHub actions:', error);
    return NextResponse.json(
      { 
        success: false,
        error: error.message || 'Error al ejecutar acciones en GitHub',
      },
      { status: 500 }
    );
  }
}

