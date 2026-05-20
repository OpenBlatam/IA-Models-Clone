/**
 * Utility types for Continuous Agent module
 * 
 * Provides helpful TypeScript utility types
 */

/**
 * Makes all properties in T readonly recursively
 */
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends (infer U)[]
    ? ReadonlyArray<DeepReadonly<U>>
    : T[P] extends object
    ? DeepReadonly<T[P]>
    : T[P];
};

/**
 * Makes all properties in T optional recursively
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends (infer U)[]
    ? DeepPartial<U>[]
    : T[P] extends object
    ? DeepPartial<T[P]>
    : T[P];
};

/**
 * Makes all properties in T required recursively
 */
export type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends (infer U)[]
    ? DeepRequired<U>[]
    : T[P] extends object
    ? DeepRequired<T[P]>
    : T[P];
};

/**
 * Extracts the return type of a function
 */
export type ReturnType<T extends (...args: unknown[]) => unknown> = T extends (
  ...args: unknown[]
) => infer R
  ? R
  : never;

/**
 * Extracts the parameter types of a function
 */
export type Parameters<T extends (...args: unknown[]) => unknown> = T extends (
  ...args: infer P
) => unknown
  ? P
  : never;

/**
 * Creates a type with only the specified keys
 */
export type Pick<T, K extends keyof T> = {
  [P in K]: T[P];
};

/**
 * Creates a type without the specified keys
 */
export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

/**
 * Makes specified keys optional
 */
export type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * Makes specified keys required
 */
export type RequiredBy<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;

/**
 * Creates a type with null and undefined excluded
 */
export type NonNullable<T> = T extends null | undefined ? never : T;

/**
 * Creates a type that represents a value or null
 */
export type Nullable<T> = T | null;

/**
 * Creates a type that represents a value or undefined
 */
export type Maybe<T> = T | undefined;

/**
 * Creates a type that represents a value, null, or undefined
 */
export type Optional<T> = T | null | undefined;

/**
 * Extracts the element type from an array
 */
export type ArrayElement<T extends readonly unknown[]> = T extends readonly (infer U)[]
  ? U
  : never;

/**
 * Creates a type from object values
 */
export type ValueOf<T> = T[keyof T];

/**
 * Creates a type from object keys
 */
export type KeyOf<T> = keyof T;

/**
 * Creates a type that represents a function that returns a promise
 */
export type AsyncFunction<T extends unknown[] = unknown[], R = unknown> = (
  ...args: T
) => Promise<R>;

/**
 * Creates a type that represents a function that may return a promise
 */
export type MaybeAsyncFunction<T extends unknown[] = unknown[], R = unknown> = (
  ...args: T
) => R | Promise<R>;

/**
 * Creates a type that represents an event handler
 */
export type EventHandler<T = unknown> = (event: T) => void | Promise<void>;

/**
 * Creates a type that represents a callback function
 */
export type Callback<T extends unknown[] = unknown[], R = void> = (...args: T) => R;

/**
 * Creates a type that represents a predicate function
 */
export type Predicate<T> = (value: T) => boolean;

/**
 * Creates a type that represents a mapper function
 */
export type Mapper<T, R> = (value: T) => R;

/**
 * Creates a type that represents a reducer function
 */
export type Reducer<T, R> = (accumulator: R, current: T) => R;

/**
 * Creates a type for React component props
 */
export type ComponentProps<T extends React.ComponentType<unknown>> = React.ComponentProps<T>;

/**
 * Creates a type for React component ref
 */
export type ComponentRef<T extends React.ComponentType<unknown>> = React.ComponentRef<T>;

/**
 * Creates a type that represents a record with string keys
 */
export type StringRecord<T> = Record<string, T>;

/**
 * Creates a type that represents a record with number keys
 */
export type NumberRecord<T> = Record<number, T>;

/**
 * Creates a type that represents a tuple of a specific length
 */
export type Tuple<T, N extends number> = N extends N
  ? number extends N
    ? T[]
    : _TupleOf<T, N, []>
  : never;

type _TupleOf<T, N extends number, R extends unknown[]> = R["length"] extends N
  ? R
  : _TupleOf<T, N, [T, ...R]>;

/**
 * Creates a type that represents a branded string
 */
export type Branded<T, B> = T & { __brand: B };

/**
 * Creates a type that represents an ID
 */
export type ID = Branded<string, "ID">;

/**
 * Creates a type that represents a timestamp
 */
export type Timestamp = Branded<number, "Timestamp">;

/**
 * Creates a type that represents a date string
 */
export type DateString = Branded<string, "DateString">;




