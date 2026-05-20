'use client';

interface AnalysisOptionsProps {
  enhance: boolean;
  advanced: boolean;
  onEnhanceChange: (value: boolean) => void;
  onAdvancedChange: (value: boolean) => void;
}

export default function AnalysisOptions({
  enhance,
  advanced,
  onEnhanceChange,
  onAdvancedChange,
}: AnalysisOptionsProps) {
  return (
    <section className="options-section">
      <div className="option-group">
        <label>
          <input
            type="checkbox"
            checked={enhance}
            onChange={(e) => onEnhanceChange(e.target.checked)}
          />
          Mejorar imagen antes de análisis
        </label>
      </div>
      <div className="option-group">
        <label>
          <input
            type="checkbox"
            checked={advanced}
            onChange={(e) => onAdvancedChange(e.target.checked)}
          />
          Usar análisis avanzado
        </label>
      </div>
    </section>
  );
}


import type { AnalysisOptions as Options } from '@/types';

interface AnalysisOptionsProps {
  options: Options;
  onChange: (options: Options) => void;
}

export default function AnalysisOptions({ options, onChange }: AnalysisOptionsProps) {
  const handleChange = (key: keyof Options) => {
    onChange({
      ...options,
      [key]: !options[key],
    });
  };

  return (
    <section className="options-section">
      <div className="option-group">
        <label>
          <input
            type="checkbox"
            checked={options.enhance}
            onChange={() => handleChange('enhance')}
          />
          Mejorar imagen antes de análisis
        </label>
      </div>
      <div className="option-group">
        <label>
          <input
            type="checkbox"
            checked={options.advanced}
            onChange={() => handleChange('advanced')}
          />
          Usar análisis avanzado
        </label>
      </div>
    </section>
  );
}






