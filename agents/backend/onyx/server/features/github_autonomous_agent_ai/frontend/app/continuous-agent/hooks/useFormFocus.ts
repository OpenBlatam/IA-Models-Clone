import { useRef, useCallback, useEffect } from "react";

type FieldRefs = {
  readonly name: React.RefObject<HTMLInputElement>;
  readonly description: React.RefObject<HTMLTextAreaElement>;
  readonly frequency: React.RefObject<HTMLInputElement>;
  readonly parameters: React.RefObject<HTMLTextAreaElement>;
  readonly goal: React.RefObject<HTMLTextAreaElement>;
};

type FieldOrder = readonly ("name" | "description" | "taskType" | "frequency" | "parameters" | "goal")[];

const FIELD_ORDER: FieldOrder = ["name", "description", "taskType", "frequency", "parameters", "goal"] as const;
const INPUT_FOCUS_DELAY = 100;
const FOCUS_FIRST_ERROR_DELAY = 100;

type UseFormFocusOptions = {
  readonly open: boolean;
  readonly errors: Record<string, string | null>;
  readonly fieldOrder?: FieldOrder;
};

export const useFormFocus = ({
  open,
  errors,
  fieldOrder = FIELD_ORDER,
}: UseFormFocusOptions) => {
  const nameInputRef = useRef<HTMLInputElement>(null);
  const descriptionInputRef = useRef<HTMLTextAreaElement>(null);
  const frequencyInputRef = useRef<HTMLInputElement>(null);
  const parametersInputRef = useRef<HTMLTextAreaElement>(null);
  const goalInputRef = useRef<HTMLTextAreaElement>(null);

  const refs: FieldRefs = {
    name: nameInputRef,
    description: descriptionInputRef,
    frequency: frequencyInputRef,
    parameters: parametersInputRef,
    goal: goalInputRef,
  };

  const focusFirstInvalidField = useCallback((): void => {
    for (const fieldName of fieldOrder) {
      const error = errors[fieldName];
      if (error !== null) {
        const ref = refs[fieldName as keyof FieldRefs];
        if (ref?.current) {
          setTimeout(() => {
            ref.current?.focus();
            ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
          }, FOCUS_FIRST_ERROR_DELAY);
        }
        return;
      }
    }
  }, [errors, fieldOrder, refs]);

  useEffect(() => {
    if (open && nameInputRef.current) {
      const timer = setTimeout(() => {
        nameInputRef.current?.focus();
      }, INPUT_FOCUS_DELAY);
      return () => clearTimeout(timer);
    }
  }, [open]);

  return {
    refs,
    focusFirstInvalidField,
  };
};





