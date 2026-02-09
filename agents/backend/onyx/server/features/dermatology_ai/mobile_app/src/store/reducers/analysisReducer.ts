import { AnalysisState, AnalysisResult, Recommendations } from '../../types';

const initialState: AnalysisState = {
  currentAnalysis: null,
  isAnalyzing: false,
  error: null,
  recommendations: null,
};

interface AnalysisAction {
  type: string;
  payload?: any;
}

const analysisReducer = (
  state: AnalysisState = initialState,
  action: AnalysisAction
): AnalysisState => {
  switch (action.type) {
    case 'ANALYSIS_START':
      return {
        ...state,
        isAnalyzing: true,
        error: null,
      };
    case 'ANALYSIS_SUCCESS':
      return {
        ...state,
        currentAnalysis: action.payload as AnalysisResult,
        isAnalyzing: false,
        error: null,
      };
    case 'ANALYSIS_FAILURE':
      return {
        ...state,
        isAnalyzing: false,
        error: action.payload as string,
      };
    case 'SET_RECOMMENDATIONS':
      return {
        ...state,
        recommendations: action.payload as Recommendations,
      };
    case 'CLEAR_ANALYSIS':
      return {
        ...state,
        currentAnalysis: null,
        recommendations: null,
      };
    default:
      return state;
  }
};

export default analysisReducer;

