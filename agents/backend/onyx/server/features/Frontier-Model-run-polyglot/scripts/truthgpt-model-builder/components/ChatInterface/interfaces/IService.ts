/**
 * Interfaces base para servicios
 * Define contratos que deben cumplir todos los servicios
 */

export interface IService<TState, TData> {
  /**
   * Agrega un elemento al estado
   */
  add(state: TState, key: string, data: TData): TState

  /**
   * Obtiene un elemento del estado
   */
  get(state: TState, key: string): TData | undefined

  /**
   * Elimina un elemento del estado
   */
  remove(state: TState, key: string): TState

  /**
   * Verifica si existe un elemento
   */
  has(state: TState, key: string): boolean
}

export interface IArrayService<TState, TData> {
  /**
   * Agrega un elemento a un array
   */
  add(state: TState, key: string, data: TData): TState

  /**
   * Elimina un elemento de un array por índice
   */
  remove(state: TState, key: string, index: number): TState

  /**
   * Obtiene todos los elementos de un array
   */
  get(state: TState, key: string): TData[]

  /**
   * Verifica si existe un array con elementos
   */
  has(state: TState, key: string): boolean
}

export interface IQueryableService<TState, TData> extends IService<TState, TData> {
  /**
   * Busca elementos que cumplan un criterio
   */
  find(state: TState, predicate: (data: TData) => boolean): TData[]

  /**
   * Filtra elementos por criterio
   */
  filter(state: TState, predicate: (data: TData) => boolean): TData[]
}



